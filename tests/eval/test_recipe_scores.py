import pytest

from datadec.recipes import DataRecipeName, RecipeNameResolver
from eval import MultiRecipeScores


def test_named_scores_resolve_official_names_and_source_aliases():
    scores = MultiRecipeScores.from_named_scores({"c4": 2.0, "Dolma1.7": 3.0})
    assert dict(scores) == {DataRecipeName.C4: 2.0, DataRecipeName.DOLMA17: 3.0}
    assert all(isinstance(recipe, DataRecipeName) for recipe in scores)


def test_multiple_aliases_cannot_overwrite_a_score():
    resolver = RecipeNameResolver.from_mapping({"first": "C4", "second": "C4"})
    with pytest.raises(ValueError, match="duplicate score for recipe: C4"):
        MultiRecipeScores.from_named_scores(
            {"first": 1.0, "second": 2.0}, resolver=resolver
        )


def test_scores_copy_the_input_and_are_read_only():
    original = {DataRecipeName.C4: 1.0}
    scores = MultiRecipeScores(original)
    original[DataRecipeName.C4] = 100.0
    assert scores[DataRecipeName.C4] == 1.0
    with pytest.raises(TypeError):
        scores.scores[DataRecipeName.C4] = 3.0


def test_internal_scores_require_canonical_enum_keys():
    with pytest.raises(TypeError, match="DataRecipeName"):
        MultiRecipeScores({"C4": 1.0})


def test_unknown_source_name_is_rejected():
    with pytest.raises(ValueError, match="unknown recipe name"):
        MultiRecipeScores.from_named_scores({"unknown": 1.0})
