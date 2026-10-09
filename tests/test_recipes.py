import pytest
from pydantic import ValidationError

from datadec.config import load_olmes_contract
from datadec.recipes import (
    DataRecipeName,
    RecipeNameResolver,
    load_recipe_name_resolver,
)


def test_default_resolver_covers_all_configured_source_names_and_official_names():
    resolver = load_recipe_name_resolver()
    assert {resolver.resolve(recipe.value) for recipe in DataRecipeName} == set(
        DataRecipeName
    )
    for alias, official in load_olmes_contract().recipe_map.items():
        assert resolver.resolve(alias) is DataRecipeName(official)


def test_aliases_allow_multiple_spellings_and_are_copied():
    aliases = {"first spelling": "Dolma1.7", "second spelling": "Dolma1.7"}
    resolver = RecipeNameResolver.from_mapping(aliases)
    aliases["first spelling"] = "C4"
    assert resolver.resolve("first spelling") is DataRecipeName.DOLMA17
    assert resolver.resolve("second spelling") is DataRecipeName.DOLMA17
    assert resolver.resolve(DataRecipeName.DOLMA17) is DataRecipeName.DOLMA17
    with pytest.raises(TypeError):
        resolver.aliases["other"] = DataRecipeName.C4


def test_aliases_load_from_toml(tmp_path):
    path = tmp_path / "recipes.toml"
    path.write_text('[recipe_map]\nfirst = "C4"\nsecond = "C4"\n')
    resolver = RecipeNameResolver.from_toml(path)
    assert resolver.resolve("first") is DataRecipeName.C4
    assert resolver.resolve("second") is DataRecipeName.C4
    assert resolver.resolve("Falcon") is DataRecipeName.FALCON


def test_unknown_names_and_alias_targets_fail():
    resolver = load_recipe_name_resolver()
    with pytest.raises(ValueError, match="unknown recipe name"):
        resolver.resolve("not a known recipe")
    with pytest.raises(ValidationError):
        RecipeNameResolver.from_mapping({"alias": "not an official name"})


def test_official_names_cannot_be_remapped():
    with pytest.raises(ValueError, match="conflicts with official recipe"):
        RecipeNameResolver.from_mapping({"C4": "Falcon"})
