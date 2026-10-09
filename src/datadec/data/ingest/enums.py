from __future__ import annotations

from enum import UNIQUE, StrEnum, verify


class ModelSizeName(StrEnum):
    M4 = "4M"
    M6 = "6M"
    M8 = "8M"
    M10 = "10M"
    M14 = "14M"
    M16 = "16M"
    M20 = "20M"
    M60 = "60M"
    M90 = "90M"
    M150 = "150M"
    M300 = "300M"
    M530 = "530M"
    M750 = "750M"
    B1 = "1B"


@verify(UNIQUE)
class DataRecipeName(StrEnum):
    DOLMA17 = "Dolma1.7"
    DOLMA17_NO_CODE = "Dolma1.7 (no code)"
    DOLMA17_NO_MATH_CODE = "Dolma1.7 (no math, code)"
    DOLMA17_NO_REDDIT = "Dolma1.7 (no Reddit)"
    DOLMA17_NO_FLAN = "Dolma1.7 (no Flan)"
    DOLMA16PP = "Dolma1.6++"
    C4 = "C4"
    FINEWEB_PRO = "FineWeb-Pro"
    FINEWEB_EDU = "FineWeb-Edu"
    FALCON = "Falcon"
    FALCON_CC = "Falcon+CC"
    FALCON_CC_QC_10 = "Falcon+CC (QC 10%)"
    FALCON_CC_QC_20 = "Falcon+CC (QC 20%)"
    FALCON_CC_QC_ORIG_10 = "Falcon+CC (QC Orig 10%)"
    FALCON_CC_QC_TULU_10 = "Falcon+CC (QC Tulu 10%)"
    DCLM_BASELINE = "DCLM-Baseline"
    DCLM_BASELINE_QC_10 = "DCLM-Baseline (QC 10%)"
    DCLM_BASELINE_QC_20 = "DCLM-Baseline (QC 20%)"
    DCLM_BASELINE_QC_7_FW2 = "DCLM-Baseline (QC 7%, FW2)"
    DCLM_BASELINE_QC_7_FW3 = "DCLM-Baseline (QC 7%, FW3)"
    DCLM_BASELINE_QC_FW_3 = "DCLM-Baseline (QC FW 3%)"
    DCLM_BASELINE_QC_FW_10 = "DCLM-Baseline (QC FW 10%)"
    MIX_DCLM_25_DOLMA_75 = "DCLM-Baseline 25% / Dolma 75%"
    MIX_DCLM_50_DOLMA_50 = "DCLM-Baseline 50% / Dolma 50%"
    MIX_DCLM_75_DOLMA_25 = "DCLM-Baseline 75% / Dolma 25%"


class Seed(StrEnum):
    DEFAULT = "default"
    SMALL_AUX_2 = "small aux 2"
    SMALL_AUX_3 = "small aux 3"
    LARGE_AUX_2 = "large aux 2"
    LARGE_AUX_3 = "large aux 3"

    @property
    def index(self) -> int:
        return _SEED_INDEX[self]


_SEED_INDEX: dict[Seed, int] = {
    Seed.DEFAULT: 0,
    Seed.SMALL_AUX_2: 1,
    Seed.SMALL_AUX_3: 2,
    Seed.LARGE_AUX_2: 3,
    Seed.LARGE_AUX_3: 4,
}


class Task(StrEnum):
    ARC_CHALLENGE = "arc_challenge"
    ARC_EASY = "arc_easy"
    BOOLQ = "boolq"
    CSQA = "csqa"
    HELLASWAG = "hellaswag"
    MMLU_ABSTRACT_ALGEBRA = "mmlu_abstract_algebra"
    MMLU_ANATOMY = "mmlu_anatomy"
    MMLU_ASTRONOMY = "mmlu_astronomy"
    MMLU_BUSINESS_ETHICS = "mmlu_business_ethics"
    MMLU_CLINICAL_KNOWLEDGE = "mmlu_clinical_knowledge"
    MMLU_COLLEGE_BIOLOGY = "mmlu_college_biology"
    MMLU_COLLEGE_CHEMISTRY = "mmlu_college_chemistry"
    MMLU_COLLEGE_COMPUTER_SCIENCE = "mmlu_college_computer_science"
    MMLU_COLLEGE_MATHEMATICS = "mmlu_college_mathematics"
    MMLU_COLLEGE_MEDICINE = "mmlu_college_medicine"
    MMLU_COLLEGE_PHYSICS = "mmlu_college_physics"
    MMLU_COMPUTER_SECURITY = "mmlu_computer_security"
    MMLU_CONCEPTUAL_PHYSICS = "mmlu_conceptual_physics"
    MMLU_ECONOMETRICS = "mmlu_econometrics"
    MMLU_ELECTRICAL_ENGINEERING = "mmlu_electrical_engineering"
    MMLU_ELEMENTARY_MATHEMATICS = "mmlu_elementary_mathematics"
    MMLU_FORMAL_LOGIC = "mmlu_formal_logic"
    MMLU_GLOBAL_FACTS = "mmlu_global_facts"
    MMLU_HIGH_SCHOOL_BIOLOGY = "mmlu_high_school_biology"
    MMLU_HIGH_SCHOOL_CHEMISTRY = "mmlu_high_school_chemistry"
    MMLU_HIGH_SCHOOL_COMPUTER_SCIENCE = "mmlu_high_school_computer_science"
    MMLU_HIGH_SCHOOL_EUROPEAN_HISTORY = "mmlu_high_school_european_history"
    MMLU_HIGH_SCHOOL_GEOGRAPHY = "mmlu_high_school_geography"
    MMLU_HIGH_SCHOOL_GOVERNMENT_AND_POLITICS = (
        "mmlu_high_school_government_and_politics"
    )
    MMLU_HIGH_SCHOOL_MACROECONOMICS = "mmlu_high_school_macroeconomics"
    MMLU_HIGH_SCHOOL_MATHEMATICS = "mmlu_high_school_mathematics"
    MMLU_HIGH_SCHOOL_MICROECONOMICS = "mmlu_high_school_microeconomics"
    MMLU_HIGH_SCHOOL_PHYSICS = "mmlu_high_school_physics"
    MMLU_HIGH_SCHOOL_PSYCHOLOGY = "mmlu_high_school_psychology"
    MMLU_HIGH_SCHOOL_STATISTICS = "mmlu_high_school_statistics"
    MMLU_HIGH_SCHOOL_US_HISTORY = "mmlu_high_school_us_history"
    MMLU_HIGH_SCHOOL_WORLD_HISTORY = "mmlu_high_school_world_history"
    MMLU_HUMAN_AGING = "mmlu_human_aging"
    MMLU_HUMAN_SEXUALITY = "mmlu_human_sexuality"
    MMLU_INTERNATIONAL_LAW = "mmlu_international_law"
    MMLU_JURISPRUDENCE = "mmlu_jurisprudence"
    MMLU_LOGICAL_FALLACIES = "mmlu_logical_fallacies"
    MMLU_MACHINE_LEARNING = "mmlu_machine_learning"
    MMLU_MANAGEMENT = "mmlu_management"
    MMLU_MARKETING = "mmlu_marketing"
    MMLU_MEDICAL_GENETICS = "mmlu_medical_genetics"
    MMLU_MISCELLANEOUS = "mmlu_miscellaneous"
    MMLU_MORAL_DISPUTES = "mmlu_moral_disputes"
    MMLU_MORAL_SCENARIOS = "mmlu_moral_scenarios"
    MMLU_NUTRITION = "mmlu_nutrition"
    MMLU_PHILOSOPHY = "mmlu_philosophy"
    MMLU_PREHISTORY = "mmlu_prehistory"
    MMLU_PROFESSIONAL_ACCOUNTING = "mmlu_professional_accounting"
    MMLU_PROFESSIONAL_LAW = "mmlu_professional_law"
    MMLU_PROFESSIONAL_MEDICINE = "mmlu_professional_medicine"
    MMLU_PROFESSIONAL_PSYCHOLOGY = "mmlu_professional_psychology"
    MMLU_PUBLIC_RELATIONS = "mmlu_public_relations"
    MMLU_SECURITY_STUDIES = "mmlu_security_studies"
    MMLU_SOCIOLOGY = "mmlu_sociology"
    MMLU_US_FOREIGN_POLICY = "mmlu_us_foreign_policy"
    MMLU_VIROLOGY = "mmlu_virology"
    MMLU_WORLD_RELIGIONS = "mmlu_world_religions"
    OPENBOOKQA = "openbookqa"
    PIQA = "piqa"
    SOCIALIQA = "socialiqa"
    WINOGRANDE = "winogrande"


MMLU_SUBJECT_TASKS: list[Task] = sorted(
    (task for task in Task if task.value.startswith("mmlu_")),
    key=lambda task: task.value,
)
