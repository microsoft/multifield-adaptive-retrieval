from typing import List, Any, Dict, Tuple, Optional
from mfar.data.typedef import Field, FieldType

"""
The below are "presets" for all the fields. They are based
on thresholds from the corpus, codified here for reproducibility
and version control.
"""
SPARSE_MAX = 1048576

MAG_FIELDS = [
    ("abstract", 512),
    ("author___affiliated_with___institution", 512),
    ("paper___cites___paper", 512),
    ("paper___has_topic___field_of_study", 64),
    ("title", 64),
]

PRIME_FIELDS = [
    ("associated with", 256),
    ("carrier", 8),
    ("contraindication", 128),
    ("details", 512),
    ("enzyme", 64),
    ("expression absent", 64),
    ("expression present", 512),
    ("indication", 32),
    ("interacts with", 512),
    ("linked to", 8),
    ("name", 64),
    ("off-label use", 8),
    ("parent-child", 256),
    ("phenotype absent", 8),
    ("phenotype present", 512),
    ("ppi", 512),
    ("side effect", 128),
    ("source", 8),
    ("synergistic interaction", 512),
    ("target", 64),
    ("transporter", 8),
    ("type", 8),
]

AMAZON_FIELDS = [
    ("also_buy", 512),
    ("also_view", 512),
    ("brand", 16),
    ("description", 512),
    ("feature", 512),
    ("qa", 512),
    ("review", 512),
    ("title", 128),
]

# What's that book dataset: unused in paper.
WTB_FIELDS = [
    ("author", 16),
    ("author_url", 64),
    ("date", 64),
    ("description", 512),
    ("genres", 64),
    ("id", 16),
    ("image_link", 64),
    ("isbn_13", 16),
    ("parsed_dates", 16),
    ("ratings", 16),
    ("reviews", 16),
    ("title", 64),
]

def generate_schema(FIELDS, dataset_name):
    fields = {}
    for field, max_seq_length in FIELDS:
        fields[f"{field}_sparse"] = Field(f"{field}_sparse", field, FieldType.SPARSE, SPARSE_MAX, dataset=dataset_name)
        fields[f"{field}_dense"] = Field(f"{field}_dense", field, FieldType.DENSE, max_seq_length, dataset=dataset_name)
    return fields

FIELDS_DICT = {
    "mag": MAG_FIELDS,
    "prime": PRIME_FIELDS,
    "amazon": AMAZON_FIELDS,
    "whatsthatbook": WTB_FIELDS
}
DATASET_NAMES = list(FIELDS_DICT.keys())
SCHEMAS = {name: generate_schema(FIELDS_DICT[name], name) for name in DATASET_NAMES}


STARK_SCHEMAS = {}
for dataset in DATASET_NAMES:
    STARK_SCHEMAS[dataset] = {
        "single_sparse": Field("single_sparse", "single", FieldType.SPARSE, SPARSE_MAX, dataset),
        "single_dense": Field("single_dense", "single", FieldType.DENSE, 512, dataset),
    }


def resolve_fields(field_names: List[str], dataset: str) -> Dict[str, Field]:
    for valid_dataset in DATASET_NAMES:
        if valid_dataset in dataset.split("/")[-1]:
            dataset_name = valid_dataset
            break
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported!")

    valid_fields = SCHEMAS[dataset_name]
    returned_dict = {}

    if isinstance(field_names, str):
        field_names = tuple([name for name in field_names.split(",")])
        field_names = [field.replace(".", " ") for field in field_names]

    for field_name in field_names:
        # Special cases first:
        if field_name == "all_sparse":
            for field_key, field in valid_fields.items():
                if field.field_type == FieldType.SPARSE:
                    returned_dict[field_key] = field
        elif field_name == "all_dense":
            for field_key, field in valid_fields.items():
                if field.field_type == FieldType.DENSE:
                    returned_dict[field_key] = field
        elif field_name == "single_sparse":
            returned_dict["single_sparse"] = STARK_SCHEMAS[dataset_name]["single_sparse"]
        elif field_name == "single_dense":
            returned_dict["single_dense"] = STARK_SCHEMAS[dataset_name]["single_dense"]
        else:
            if field_name not in valid_fields:
                raise ValueError(f"Field {field_name} not found in dataset {dataset}")
            returned_dict[field_name] = valid_fields[field_name]

    # Always sort dict keys for consistency - dense first then sparse
    sorted_keys = sorted(returned_dict.keys())
    dense_keys = [k for k in sorted_keys if returned_dict[k].field_type == FieldType.DENSE]
    sparse_keys = [k for k in sorted_keys if returned_dict[k].field_type == FieldType.SPARSE]
    return {k: returned_dict[k] for k in dense_keys + sparse_keys}