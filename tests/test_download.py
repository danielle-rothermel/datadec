from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock, call, patch
from urllib.request import Request

import pandas as pd
import pytest

from datadec.config import (
    DetailFile,
    PublishedResultFile,
    PublishedResultsManifest,
    SourceManifest,
    load_published_results_manifest,
    load_source_manifest,
)
from datadec.data import download
from datadec.data.download import download_sources
from datadec.data.selection import (
    resolve_olmes_detail_recipes,
    resolve_selection,
    selected_published_result_sources,
)
from datadec.data.artifacts import DataArtifacts

PUBLISHED_RESULTS_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1weYlEOlHrA_fzT2OsRa40uLc4EKTGz1D"
)
DETAIL_BYTES = b"verified detail archive"


def _write_dataset_parquet(path: Path) -> None:
    pd.DataFrame({"value": [1]}).to_parquet(path, index=False)


class FakeResponse:
    def __init__(
        self,
        *chunks: bytes,
        status: int = 200,
        headers: dict[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._chunks = list(chunks)
        self._error = error

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        if self._error is not None:
            error = self._error
            self._error = None
            raise error
        return b""


def published_file(
    *,
    path: str = "outputs2/1_metric_transformed.csv",
    expected_size: int = 6,
    category: str = "published_results",
    file_id: str = "drive-file-id",
    content: bytes | None = None,
) -> PublishedResultFile:
    structured = category == "published_results"
    expected_content = content or (
        b"abcdef" if expected_size == 6 else b"x" * expected_size
    )
    return PublishedResultFile.model_validate(
        {
            "id": file_id,
            "path": path,
            "expected_size": expected_size,
            "sha256": hashlib.sha256(expected_content).hexdigest(),
            "category": category,
            "publication_unit": "outputs2" if structured else None,
            "schema": "transformed" if structured else None,
        }
    )


def detail_manifest(*recipes: str) -> SourceManifest:
    manifest = load_source_manifest()
    files = tuple(
        DetailFile(
            recipe=recipe,
            expected_size=len(DETAIL_BYTES),
            sha256=hashlib.sha256(DETAIL_BYTES).hexdigest(),
        )
        for recipe in recipes
    )
    return manifest.model_copy(
        update={
            "olmes_details": manifest.olmes_details.model_copy(update={"files": files})
        }
    )


def write_detail_download(**kwargs: object) -> Path:
    destination = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(DETAIL_BYTES)
    return destination


def test_data_paths_use_exact_root_and_raw_outputs(tmp_path: Path) -> None:
    paths = DataArtifacts(tmp_path)

    assert paths.data_dir == tmp_path
    assert paths.get_path("ppl_raw") == tmp_path / "raw/ppl.parquet"
    assert paths.get_path("dwn_raw") == tmp_path / "raw/olmes.parquet"
    assert paths.get_path("ppl_processed") == tmp_path / "processed/ppl.parquet"
    assert paths.get_path("olmes_processed") == tmp_path / "processed/olmes.parquet"


def test_download_sources_preserves_ppl_olmes_then_detail_order(
    tmp_path: Path,
) -> None:
    paths = DataArtifacts(tmp_path)
    ppl_dataset = MagicMock()
    olmes_dataset = MagicMock()
    ppl_dataset.to_parquet.side_effect = _write_dataset_parquet
    olmes_dataset.to_parquet.side_effect = _write_dataset_parquet
    with (
        patch(
            "datadec.data.download.load_dataset",
            side_effect=[ppl_dataset, olmes_dataset],
        ) as load_dataset,
        patch(
            "datadec.data.download.hf_hub_download",
            side_effect=write_detail_download,
        ) as hf_hub_download,
    ):
        results = download_sources(
            paths,
            resolve_selection(
                ppl=True,
                olmes=True,
                olmes_details=["fineweb-pro", "c4", "fineweb-pro"],
            ),
            manifest=detail_manifest("fineweb-pro", "c4"),
        )

    cache_dir = tmp_path / "cache/huggingface"
    assert [result.source for result in results] == [
        "ppl",
        "olmes",
        "olmes-details:fineweb-pro",
        "olmes-details:c4",
    ]
    assert [result.status for result in results] == ["downloaded"] * 4
    assert load_dataset.call_args_list == [
        call(
            "allenai/DataDecide-ppl-results",
            split="train",
            revision="c4a9fa360a0c8351e71f3ede04dd165995fab68c",
            cache_dir=cache_dir,
        ),
        call(
            "allenai/DataDecide-eval-results",
            split="train",
            revision="9919b5a0e61e57a85021263918fa82d6ceaee038",
            cache_dir=cache_dir,
        ),
    ]
    ppl_temporary = ppl_dataset.to_parquet.call_args.args[0]
    olmes_temporary = olmes_dataset.to_parquet.call_args.args[0]
    assert ppl_temporary.parent == tmp_path / "raw"
    assert olmes_temporary.parent == tmp_path / "raw"
    assert ppl_temporary.name.startswith(".ppl.parquet.")
    assert olmes_temporary.name.startswith(".olmes.parquet.")
    assert not ppl_temporary.exists()
    assert not olmes_temporary.exists()
    assert pd.read_parquet(tmp_path / "raw/ppl.parquet")["value"].tolist() == [1]
    assert pd.read_parquet(tmp_path / "raw/olmes.parquet")["value"].tolist() == [1]
    assert hf_hub_download.call_args_list == [
        call(
            repo_id="allenai/DataDecide-eval-instances",
            repo_type="dataset",
            filename="models/fineweb-pro.tar.gz",
            revision="23f3b2e186ca6c39026e3efa00e4af397680c075",
            cache_dir=cache_dir,
            local_dir=tmp_path / "raw/olmes-details",
            force_download=False,
        ),
        call(
            repo_id="allenai/DataDecide-eval-instances",
            repo_type="dataset",
            filename="models/c4.tar.gz",
            revision="23f3b2e186ca6c39026e3efa00e4af397680c075",
            cache_dir=cache_dir,
            local_dir=tmp_path / "raw/olmes-details",
            force_download=False,
        ),
    ]


def test_existing_outputs_are_reused_without_network_calls(tmp_path: Path) -> None:
    paths = DataArtifacts(tmp_path)
    outputs = [
        tmp_path / "raw/ppl.parquet",
        tmp_path / "raw/olmes.parquet",
        tmp_path / "raw/olmes-details/models/c4.tar.gz",
    ]
    for output in outputs[:2]:
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_dataset_parquet(output)
    for output in outputs[2:]:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(DETAIL_BYTES)

    with (
        patch("datadec.data.download.load_dataset") as load_dataset,
        patch("datadec.data.download.hf_hub_download") as hf_hub_download,
    ):
        results = download_sources(
            paths,
            resolve_selection(ppl=True, olmes=True, olmes_details=["c4"]),
            manifest=detail_manifest("c4"),
        )

    assert [result.destination for result in results] == outputs
    assert [result.status for result in results] == ["reused"] * 3
    load_dataset.assert_not_called()
    hf_hub_download.assert_not_called()


def test_force_propagates_to_dataset_and_detail_downloads(tmp_path: Path) -> None:
    paths = DataArtifacts(tmp_path)
    dataset = MagicMock()
    dataset.to_parquet.side_effect = _write_dataset_parquet
    with (
        patch("datadec.data.download.load_dataset", return_value=dataset) as load,
        patch(
            "datadec.data.download.hf_hub_download",
            side_effect=write_detail_download,
        ) as hf_download,
    ):
        download_sources(
            paths,
            resolve_selection(ppl=True, olmes_details=["c4"]),
            force=True,
            manifest=detail_manifest("c4"),
        )

    load.assert_called_once_with(
        "allenai/DataDecide-ppl-results",
        split="train",
        revision="c4a9fa360a0c8351e71f3ede04dd165995fab68c",
        cache_dir=tmp_path / "cache/huggingface",
        download_mode="force_redownload",
    )
    hf_download.assert_called_once_with(
        repo_id="allenai/DataDecide-eval-instances",
        repo_type="dataset",
        filename="models/c4.tar.gz",
        revision="23f3b2e186ca6c39026e3efa00e4af397680c075",
        cache_dir=tmp_path / "cache/huggingface",
        local_dir=tmp_path / "raw/olmes-details",
        force_download=True,
    )


def test_invalid_existing_dataset_output_is_redownloaded(tmp_path: Path) -> None:
    paths = DataArtifacts(tmp_path)
    output = paths.get_path("ppl_raw")
    output.parent.mkdir(parents=True)
    output.touch()
    dataset = MagicMock()
    dataset.to_parquet.side_effect = _write_dataset_parquet

    with patch("datadec.data.download.load_dataset", return_value=dataset):
        result = download_sources(paths, resolve_selection(ppl=True))

    assert result[0].status == "downloaded"
    assert pd.read_parquet(output)["value"].tolist() == [1]


def test_existing_detail_archive_requires_matching_digest(tmp_path: Path) -> None:
    destination = tmp_path / "raw/olmes-details/models/c4.tar.gz"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"x" * len(DETAIL_BYTES))

    with (
        patch("datadec.data.download.hf_hub_download") as hf_hub_download,
        pytest.raises(ValueError, match="unexpected SHA-256"),
    ):
        download_sources(
            DataArtifacts(tmp_path),
            resolve_selection(olmes_details=["c4"]),
            manifest=detail_manifest("c4"),
        )

    hf_hub_download.assert_not_called()


def test_downloaded_detail_archive_requires_matching_digest(tmp_path: Path) -> None:
    def write_invalid_detail(**kwargs: object) -> Path:
        destination = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x" * len(DETAIL_BYTES))
        return destination

    with (
        patch(
            "datadec.data.download.hf_hub_download",
            side_effect=write_invalid_detail,
        ),
        pytest.raises(ValueError, match="unexpected SHA-256"),
    ):
        download_sources(
            DataArtifacts(tmp_path),
            resolve_selection(olmes_details=["c4"]),
            manifest=detail_manifest("c4"),
        )


def test_failed_forced_dataset_export_preserves_existing_output(
    tmp_path: Path,
) -> None:
    paths = DataArtifacts(tmp_path)
    output = paths.get_path("ppl_raw")
    output.parent.mkdir(parents=True)
    pd.DataFrame({"value": [7]}).to_parquet(output, index=False)
    dataset = MagicMock()

    def fail_after_partial_write(path: Path) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("interrupted export")

    dataset.to_parquet.side_effect = fail_after_partial_write
    with (
        patch("datadec.data.download.load_dataset", return_value=dataset),
        pytest.raises(RuntimeError, match="interrupted export"),
    ):
        download_sources(paths, resolve_selection(ppl=True), force=True)

    assert pd.read_parquet(output)["value"].tolist() == [7]
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_all_detail_recipes_use_config_order() -> None:
    source = load_source_manifest().olmes_details

    assert resolve_olmes_detail_recipes(["all"], source) == tuple(source.recipes)
    assert resolve_olmes_detail_recipes(["c4", "all", "c4"], source) == tuple(
        source.recipes
    )


def test_download_sources_uses_only_the_canonical_selection_api() -> None:
    parameters = inspect.signature(download_sources).parameters

    assert tuple(parameters)[:2] == ("paths", "selection")
    assert "ppl" not in parameters
    assert "olmes" not in parameters


def test_verbose_status_identifies_source_destination_and_reuse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "raw/ppl.parquet"
    output.parent.mkdir(parents=True)
    _write_dataset_parquet(output)

    download_sources(DataArtifacts(tmp_path), resolve_selection(ppl=True), verbose=True)

    assert capsys.readouterr().out == f"ppl: reused -> {output}\n"


def test_download_implementation_does_not_use_snapshot_download() -> None:
    assert "snapshot_download" not in inspect.getsource(download)


def test_published_result_destinations_preserve_category_mapping(
    tmp_path: Path,
) -> None:
    paths = DataArtifacts(tmp_path)
    raw = published_file(path="raw_data/results_ladder.csv", category="scaling_law")
    reference = published_file(
        path="per_task_out/arc/figure.pdf", category="published_figures"
    )

    assert paths.published_result_download_path(raw) == (
        tmp_path / "raw/scaling-law/results_ladder.csv"
    )
    assert paths.published_result_download_path(reference) == (
        tmp_path / "reference/published-results/per_task_out/arc/figure.pdf"
    )


def test_fresh_published_result_download_streams_and_atomically_completes(
    tmp_path: Path,
) -> None:
    source = published_file()
    response = FakeResponse(b"abc", b"def")
    with patch("datadec.data.download.urlopen", return_value=response) as open_url:
        result = download._download_published_result_file(
            DataArtifacts(tmp_path), source, force=False
        )

    destination = (
        tmp_path / "reference/published-results/outputs2/1_metric_transformed.csv"
    )
    request = open_url.call_args.args[0]
    assert isinstance(request, Request)
    assert request.full_url == (
        "https://drive.usercontent.google.com/download?id=drive-file-id"
        "&export=download&confirm=t"
    )
    assert open_url.call_args.kwargs == {"timeout": download._DOWNLOAD_TIMEOUT_SECONDS}
    assert request.get_header("Range") is None
    assert destination.read_bytes() == b"abcdef"
    assert not destination.with_name("1_metric_transformed.csv.part").exists()
    assert result == download.DownloadResult(
        "published-results:outputs2/1_metric_transformed.csv",
        destination,
        "downloaded",
    )


def test_published_result_download_resumes_valid_partial_with_range(
    tmp_path: Path,
) -> None:
    source = published_file()
    destination = (
        tmp_path / "reference/published-results/outputs2/1_metric_transformed.csv"
    )
    partial = destination.with_name("1_metric_transformed.csv.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"abc")
    response = FakeResponse(
        b"def",
        status=206,
        headers={"Content-Range": "bytes 3-5/6"},
    )

    with patch("datadec.data.download.urlopen", return_value=response) as open_url:
        download._download_published_result_file(
            DataArtifacts(tmp_path), source, force=False
        )

    request = open_url.call_args.args[0]
    assert request.get_header("Range") == "bytes=3-"
    assert destination.read_bytes() == b"abcdef"
    assert not partial.exists()


def test_published_result_download_restarts_when_server_ignores_range(
    tmp_path: Path,
) -> None:
    source = published_file()
    destination = (
        tmp_path / "reference/published-results/outputs2/1_metric_transformed.csv"
    )
    partial = destination.with_name("1_metric_transformed.csv.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"old")

    with patch(
        "datadec.data.download.urlopen",
        return_value=FakeResponse(b"abcdef", status=200),
    ):
        download._download_published_result_file(
            DataArtifacts(tmp_path), source, force=False
        )

    assert destination.read_bytes() == b"abcdef"


def test_published_result_reuses_expected_size_and_force_redownloads(
    tmp_path: Path,
) -> None:
    source = published_file()
    destination = (
        tmp_path / "reference/published-results/outputs2/1_metric_transformed.csv"
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"abcdef")

    with patch("datadec.data.download.urlopen") as open_url:
        reused = download._download_published_result_file(
            DataArtifacts(tmp_path), source, force=False
        )
    open_url.assert_not_called()
    assert reused.status == "reused"

    with patch(
        "datadec.data.download.urlopen",
        return_value=FakeResponse(b"forced"),
    ):
        source = published_file(content=b"forced")
        forced = download._download_published_result_file(
            DataArtifacts(tmp_path), source, force=True
        )
    assert forced.status == "downloaded"
    assert destination.read_bytes() == b"forced"


def test_published_result_rejects_mismatched_complete_file(
    tmp_path: Path,
) -> None:
    source = published_file()
    destination = (
        tmp_path / "reference/published-results/outputs2/1_metric_transformed.csv"
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"wrong")

    with patch("datadec.data.download.urlopen") as open_url:
        with pytest.raises(ValueError, match="existing file.*unexpected size"):
            download._download_published_result_file(
                DataArtifacts(tmp_path), source, force=False
            )

    open_url.assert_not_called()
    assert destination.read_bytes() == b"wrong"


def test_published_result_rejects_same_size_wrong_digest(tmp_path: Path) -> None:
    source = published_file()
    destination = (
        tmp_path / "reference/published-results/outputs2/1_metric_transformed.csv"
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stored")

    with (
        patch("datadec.data.download.urlopen") as open_url,
        pytest.raises(ValueError, match="unexpected SHA-256"),
    ):
        download._download_published_result_file(
            DataArtifacts(tmp_path), source, force=False
        )

    open_url.assert_not_called()


def test_published_result_rejects_complete_partial_with_wrong_digest(
    tmp_path: Path,
) -> None:
    source = published_file()
    destination = (
        tmp_path / "reference/published-results/outputs2/1_metric_transformed.csv"
    )
    partial = destination.with_name(f"{destination.name}.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"stored")

    with (
        patch("datadec.data.download.urlopen") as open_url,
        pytest.raises(ValueError, match="unexpected SHA-256"),
    ):
        download._download_published_result_file(
            DataArtifacts(tmp_path), source, force=False
        )

    open_url.assert_not_called()
    assert not destination.exists()
    assert partial.read_bytes() == b"stored"


def test_published_result_download_digest_mismatch_keeps_partial(
    tmp_path: Path,
) -> None:
    source = published_file(content=b"ghijkl")
    with patch("datadec.data.download.urlopen", return_value=FakeResponse(b"abcdef")):
        with pytest.raises(RuntimeError, match="failed to download") as exc_info:
            download._download_published_result_file(
                DataArtifacts(tmp_path), source, force=False
            )

    destination = (
        tmp_path / "reference/published-results/outputs2/1_metric_transformed.csv"
    )
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "unexpected SHA-256" in str(exc_info.value.__cause__)
    assert not destination.exists()
    assert destination.with_name(f"{destination.name}.part").read_bytes() == b"abcdef"


def test_published_result_size_mismatch_keeps_only_partial_file(
    tmp_path: Path,
) -> None:
    source = published_file()
    with patch("datadec.data.download.urlopen", return_value=FakeResponse(b"short")):
        with pytest.raises(RuntimeError, match="failed to download") as exc_info:
            download._download_published_result_file(
                DataArtifacts(tmp_path), source, force=False
            )

    destination = (
        tmp_path / "reference/published-results/outputs2/1_metric_transformed.csv"
    )
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert not destination.exists()
    assert (
        destination.with_name("1_metric_transformed.csv.part").read_bytes() == b"short"
    )


def test_published_result_failure_preserves_cause_and_completed_destination(
    tmp_path: Path,
) -> None:
    source = published_file()
    destination = (
        tmp_path / "reference/published-results/outputs2/1_metric_transformed.csv"
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stored")
    network_error = OSError("connection lost")
    response = FakeResponse(b"new", error=network_error)

    with patch("datadec.data.download.urlopen", return_value=response):
        with pytest.raises(RuntimeError) as exc_info:
            download._download_published_result_file(
                DataArtifacts(tmp_path), source, force=True
            )

    assert exc_info.value.__cause__ is network_error
    assert destination.read_bytes() == b"stored"
    assert destination.with_name("1_metric_transformed.csv.part").read_bytes() == b"new"


def test_drive_selectors_are_disjoint_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    files = (
        published_file(
            path="raw_data/second.csv",
            expected_size=1,
            category="scaling_law",
            file_id="raw-2",
        ),
        published_file(
            path="outputs2/1_primary_transformed.csv",
            expected_size=1,
            file_id="published-2",
        ),
        published_file(
            path="raw_data/first.csv",
            expected_size=1,
            category="scaling_law",
            file_id="raw-1",
        ),
        published_file(
            path="outputs2/1_metric_transformed.csv",
            expected_size=1,
            file_id="published-1",
        ),
        published_file(
            path="per_task_out/arc/figure.pdf",
            expected_size=1,
            category="published_figures",
            file_id="figure-1",
        ),
    )
    manifest = PublishedResultsManifest(
        folder_url=PUBLISHED_RESULTS_FOLDER_URL, files=files
    )
    for file in files:
        destination = DataArtifacts(tmp_path).published_result_download_path(file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x")

    scaling = download_sources(
        DataArtifacts(tmp_path),
        resolve_selection(scaling_law=True),
        published_results_manifest=manifest,
    )
    published = download_sources(
        DataArtifacts(tmp_path),
        resolve_selection(units=("outputs2",)),
        published_results_manifest=manifest,
    )
    combined = download_sources(
        DataArtifacts(tmp_path),
        resolve_selection(scaling_law=True, units=("outputs2",)),
        published_results_manifest=manifest,
    )
    figures = download_sources(
        DataArtifacts(tmp_path),
        resolve_selection(published_figures=True),
        published_results_manifest=manifest,
    )

    assert [result.source for result in scaling] == [
        "scaling-law:raw_data/second.csv",
        "scaling-law:raw_data/first.csv",
    ]
    assert [result.source for result in published] == [
        "published-results:outputs2/1_primary_transformed.csv",
        "published-results:outputs2/1_metric_transformed.csv",
    ]
    assert [result.source for result in combined] == [
        *[result.source for result in scaling],
        *[result.source for result in published],
    ]
    assert len({result.destination for result in combined}) == 4
    assert [result.source for result in figures] == [
        "published-figures:per_task_out/arc/figure.pdf"
    ]
    assert {result.destination for result in combined}.isdisjoint(
        result.destination for result in figures
    )


def test_real_manifest_drive_selectors_pin_counts_bytes_and_use_no_network(
    tmp_path: Path,
) -> None:
    manifest = load_published_results_manifest()

    def selected_result(
        paths: DataArtifacts, source: PublishedResultFile, *, force: bool
    ) -> download.DownloadResult:
        assert force is False
        return download.DownloadResult(
            source.category,
            paths.published_result_download_path(source),
            "reused",
        )

    with (
        patch(
            "datadec.data.download._download_published_result_file",
            side_effect=selected_result,
        ) as download_file,
        patch("datadec.data.download.urlopen") as urlopen,
    ):
        structured = download_sources(
            DataArtifacts(tmp_path),
            resolve_selection(published_results=True),
            published_results_manifest=manifest,
        )
        figures = download_sources(
            DataArtifacts(tmp_path),
            resolve_selection(published_figures=True),
            published_results_manifest=manifest,
        )

    structured_sources = [
        call.args[1] for call in download_file.call_args_list[: len(structured)]
    ]
    figure_sources = [
        call.args[1] for call in download_file.call_args_list[len(structured) :]
    ]
    assert len(structured) == 51
    assert len(figures) == 80
    assert sum(source.expected_size for source in structured_sources) == 8_974_174_175
    assert sum(source.expected_size for source in figure_sources) == 83_178_155
    assert {source.category for source in structured_sources} == {"published_results"}
    assert {source.category for source in figure_sources} == {"published_figures"}
    urlopen.assert_not_called()


def test_raw_published_results_downloads_every_source_in_selected_schema_families(
    tmp_path: Path,
) -> None:
    manifest = load_published_results_manifest()

    def selected_result(
        paths: DataArtifacts, source: PublishedResultFile, *, force: bool
    ) -> download.DownloadResult:
        return download.DownloadResult(
            source.path,
            paths.published_result_download_path(source),
            "reused",
        )

    with patch(
        "datadec.data.download._download_published_result_file",
        side_effect=selected_result,
    ) as download_file:
        results = download_sources(
            DataArtifacts(tmp_path),
            resolve_selection(units=["outputs2"]),
            published_results_manifest=manifest,
        )

    selected_sources = [item.args[1] for item in download_file.call_args_list]
    expected_sources = selected_published_result_sources(("outputs2",), manifest)
    assert selected_sources == list(expected_sources)
    assert {source.schema for source in expected_sources} == {
        "transformed",
        "prediction_model_scale",
        "target_pairs",
    }
    assert {source.publication_unit for source in expected_sources} > {"outputs2"}
    assert len(results) == len(expected_sources)
