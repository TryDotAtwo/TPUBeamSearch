MXU_DIM = 128
TPU_ROW_ALIGNMENT = 8
TPU_COL_ALIGNMENT = 128
TPU_BF16_VECTOR_ALIGNMENT = 256


def pad_to_multiple(value: int, alignment: int) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def validate_matrix_tile(*, bm: int, bk: int, bn: int) -> None:
    if bm <= 0 or bm % TPU_ROW_ALIGNMENT:
        raise ValueError(f"bm must be a positive multiple of {TPU_ROW_ALIGNMENT}")
    if bk <= 0 or bk % TPU_COL_ALIGNMENT:
        raise ValueError(f"bk must be a positive multiple of {TPU_COL_ALIGNMENT}")
    if bn <= 0 or bn % TPU_COL_ALIGNMENT:
        raise ValueError(f"bn must be a positive multiple of {TPU_COL_ALIGNMENT}")

