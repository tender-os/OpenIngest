# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
AZURE_DOCUMENT_INTELLIGENCE_KEY = "AZURE_DOCUMENT_INTELLIGENCE_KEY"


### OUTPUT API FROM THE WORKFLOW #####
class PageFragmentType(str, Enum):
    """
    Type of a page fragment.
    """

    SECTION_HEADER = "section_header"
    TITLE = "title"

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    FORMULA = "formula"
    FORM = "form"
    KEY_VALUE_REGION = "key_value_region"
    DOCUMENT_INDEX = "document_index"
    LIST_ITEM = "list_item"

    TABLE_CAPTION = "table_caption"
    FIGURE_CAPTION = "figure_caption"
    FORMULA_CAPTION = "formula_caption"

    PAGE_FOOTER = "page_footer"
    PAGE_HEADER = "page_header"
    PAGE_NUMBER = "page_number"
    SIGNATURE = "signature"

    TRACKED_CHANGES = "tracked_changes"
    COMMENTS = "comments"
    BARCODE = "barcode"


def fragment_type_from_doclayout_yolo_class(name: str) -> PageFragmentType:
    if name == "title":
        return PageFragmentType.SECTION_HEADER
    elif name in ["abandon", "plain text"]:
        return PageFragmentType.TEXT
    elif name == "figure":
        return PageFragmentType.FIGURE
    elif name == "figure_caption":
        return PageFragmentType.FIGURE_CAPTION
    elif name == "table":
        return PageFragmentType.TABLE
    elif name in ["table_caption", "table_footnote"]:
        return PageFragmentType.TABLE_CAPTION
    elif name == "isolate_formula":
        return PageFragmentType.FORMULA
    elif name == "formula_caption":
        return PageFragmentType.FORMULA_CAPTION
    elif name == "document_index":
        return PageFragmentType.DOCUMENT_INDEX
    elif name == "form":
        return PageFragmentType.FORM
    elif name == "list_item":
        return PageFragmentType.LIST_ITEM
    else:
        raise ValueError(f"Unknown fragment type: {name}")


def fragment_type_from_doclaynet_class(name: str) -> PageFragmentType:
    if name == "Caption":
        return PageFragmentType.FIGURE_CAPTION
    elif name in ["Footnote"]:
        return PageFragmentType.FIGURE_CAPTION
    elif name == "Formula":
        return PageFragmentType.FORMULA
    elif name == "List-item":
        return PageFragmentType.LIST_ITEM
    elif name == "Page-footer":
        return PageFragmentType.PAGE_FOOTER
    elif name == "Page-header":
        return PageFragmentType.PAGE_HEADER
    elif name == "Picture":
        return PageFragmentType.FIGURE
    elif name == "Page-number":
        return PageFragmentType.FORM
    elif name in ["Section-header"]:
        return PageFragmentType.SECTION_HEADER
    elif name == "Table":
        return PageFragmentType.TABLE
    elif name == "Text":
        return PageFragmentType.TEXT
    elif name == "Title":
        return PageFragmentType.TITLE
    elif name == "Page-number":
        return PageFragmentType.PAGE_NUMBER
    else:
        raise ValueError(f"Unknown fragment type: {name}")


def fragment_type_from_d4la_class(name: str) -> PageFragmentType:
    if name == "Section-header":
        return PageFragmentType.SECTION_HEADER
    if name == "Text":
        return PageFragmentType.TEXT
    if name == "RegionKV":
        return PageFragmentType.KEY_VALUE_REGION
    if name == "Table":
        return PageFragmentType.TABLE
    if name == "Formula":
        return PageFragmentType.FORMULA
    if name == "Figure":
        return PageFragmentType.FIGURE
    if name == "pageFooter":
        return PageFragmentType.PAGE_FOOTER
    if name == "pageNumber":
        return PageFragmentType.PAGE_NUMBER
    if name == "Signature":
        return PageFragmentType.SIGNATURE
    raise ValueError(f"Unknown fragment type: {name}")


class Text(BaseModel):
    content: str
    html: Optional[str] = None


class SectionHeader(BaseModel):
    content: str
    level: int  # 1 for #, 2 for ##, 3 for ###, etc.


class Signature(BaseModel):
    content: Optional[str] = None


class TableCell(BaseModel):
    text: str
    bounding_box: dict[str, float]
    ref_id: Optional[str] = None  # Format: page.reading_order.cell_index


class ListItem(BaseModel):
    content: str


class Table(BaseModel):
    content: str
    summary: Optional[str] = None
    cells: List[TableCell]
    html: Optional[str] = None
    markdown: Optional[str] = None
    table_checked: Optional[bool] = False  # Checked and processed through table correction


class Figure(BaseModel):
    content: str
    summary: Optional[str] = None
    image_base64: Optional[str] = None  # Base64-encoded image data for rendering
    text_bounding_boxes: Optional[List[TableCell]] = None


class Chart(BaseModel):
    content: str
    image_base64: Optional[str] = None  # Base64-encoded image data for rendering


class PageFragment(BaseModel):
    fragment_type: PageFragmentType
    content: Union[Text, Table, Figure, Chart, ListItem, Signature, SectionHeader]
    reading_order: Optional[int] = None
    bbox: Optional[dict[str, float]] = None
    ref_id: Optional[str] = None  # Format: page.reading_order or page.reading_order.cell_index


class Page(BaseModel):
    """
    Page in a document.
    """

    page_number: int
    page_fragments: Optional[List[PageFragment]] = []
    dimensions: Optional[Tuple[int, int]] = None
    page_dimensions: Optional[Dict[str, int]] = None
    page_class: Optional[Union[str, List[str]]] = None
    classification_reason: Optional[str] = None
    classification_confidence: Optional[float] = None


class StructuredData(BaseModel):
    data: Optional[Any]
    page_numbers: Union[List[int], int]
    schema_name: Optional[str] = None


class Chunk(BaseModel):
    content: str
    page_number: int  # For backward compatibility - the starting page
    page_numbers: Optional[List[int]] = None  # All pages this chunk spans
    element_ids: Optional[List[str]] = (
        None  # ref_ids of elements in this chunk (e.g., ["2.5", "2.6", "3.1"])
    )


class PageClass(BaseModel):
    page_numbers: List[int]
    classification_reasons: Optional[Dict[int, str]] = None
    classification_confidences: Optional[Dict[int, float]] = None
    page_class: str


class MergeTableActions(BaseModel):
    pages: List[int]
    ref_ids: Optional[List[str]] = None
    target_columns: Optional[int] = None


class MergedTable(BaseModel):
    merged_table_id: str
    merged_table_html: str
    start_page: int
    end_page: int
    pages_merged: int
    summary: Optional[str] = None
    merge_actions: MergeTableActions


class ParsedDocument(BaseModel):
    parsed_pages_count: Optional[int] = None
    pages: Optional[List[Page]] = None
    chunks: List[Chunk]
    merged_tables: Optional[List[MergedTable]] = None
    structured_data: Optional[List[StructuredData]] = None
    total_pages: Optional[int] = None
    page_classes: Optional[List[PageClass]] = None
    docx_converted_pdf_base64: Optional[str] = None  # For DOCX files converted to PDF
    document_markdown: Optional[str] = None  # Full document markdown representation
    filled_pdf_base64: Optional[str] = None
    filled_docx_base64: Optional[str] = None
    form_filling_metadata: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    pages_parsed: int
    signature_detection: bool = Field(default=False)
    ocr_input_tokens_used: Optional[int] = None
    ocr_output_tokens_used: Optional[int] = None
    extraction_input_tokens_used: Optional[int] = None
    extraction_output_tokens_used: Optional[int] = None
    summarization_input_tokens_used: Optional[int] = None
    summarization_output_tokens_used: Optional[int] = None
    header_correction_input_tokens_used: Optional[int] = None
    header_correction_output_tokens_used: Optional[int] = None


class ParsedDocumentRef(BaseModel):
    document: Optional[Dict] = None
    usage: Optional[Usage] = None


class FormFillingRequest(BaseModel):
    fill_prompt: Optional[str] = None
    ignore_source_values: Optional[bool] = False
    no_acroform: Optional[bool] = False
    no_widget_detection: Optional[bool] = False


class QuotaResourceType(str, Enum):
    PAGES_PARSED = "pages_parsed"
    SIGNATURE_DETECTION = "signature_detection"


class ResourceQuotaRequest(BaseModel):
    # this is to allow the use of alias for the fields
    model_config = ConfigDict(populate_by_name=True)

    resource_type: QuotaResourceType = Field(alias="resourceType")
    remaining_quota: int = Field(
        alias="remainingQuota",
        description="Remaining quota for this resource type. Use -1 for unlimited quota.",
    )


class OrganizationQuotaRequest(BaseModel):
    # this is to allow the use of alias for the fields
    model_config = ConfigDict(populate_by_name=True)

    organization_id: str = Field(alias="organizationId")
    quotas: List[ResourceQuotaRequest]


##### REQUEST API INTO THE WORKFLOW #####
class PageClassDefinition(BaseModel):
    class_name: str
    description: str


class ClassificationRequest(BaseModel):
    """
    Request for page classification.

    classification_type:
        - "multi-label": Each page can belong to multiple classes simultaneously (default).
        - "multi-class": Each page can belong to only one class.
    """

    class_definitions: List[PageClassDefinition]
    classification_type: Literal["multi_label", "multi_class"] = Field(
        default="multi_label",
        description=(
            "Type of classification to perform. "
            "'multi-label' allows each page to have multiple classes. "
            "'multi-class' restricts each page to a single class."
        ),
    )


# Update partition strategy
class PatternChunking(BaseModel):
    strategy: Literal["patterns"]
    start_patterns: Optional[List[str]] = None
    end_patterns: Optional[List[str]] = None


PartitionStrategy = Union[
    Literal["none"], Literal["page"], Literal["section"], Literal["fragment"], PatternChunking
]


class StructuredExtractionRequest(BaseModel):
    json_schema: str
    prompt: Optional[str] = None
    model_provider: Optional[str] = None
    skip_ocr: Optional[bool] = False
    chunking_strategy: Optional[PartitionStrategy] = None
    page_classes: Optional[List[str]] = None
    schema_name: str
    enable_citation: Optional[bool] = False


class ParseRequest(BaseModel):
    file_bytes: Optional[str] = None
    file_url: Optional[str] = None

    pages_to_parse: Optional[List[int]] = None
    file_name: str
    mime_type: str
    skew_correction: bool = False
    detect_signature: bool = False
    detect_barcode: bool = False
    debug: bool = False
    chunk_strategy: Optional[str] = None
    table_parsing_strategy: Optional[Literal["tsr", "vlm"]] = "vlm"
    table_output_mode: Optional[Literal["html", "json", "markdown"]] = "markdown"
    ocr_model: Optional[Literal["dots-ocr", "azure-di", "textract", "gemini"]] = "azure-di"
    page_classification_request: Optional[ClassificationRequest] = None
    structured_extraction_requests: Optional[List[StructuredExtractionRequest]] = None
    disable_layout_detection: Optional[bool] = False
    # When True, the OCR backend returns the OCR'd ParseResult and the pipeline
    # stops before the post-OCR DAG (TableMerging / classify / extract / format).
    # Used by the GPU/CPU split so the GPU stage only ever runs OCR; the post-OCR
    # stage resumes via ``resume_post_ocr_app``. Default False = unchanged behaviour.
    ocr_only: bool = False
    # When True (used with ocr_only), DotsOCRTask stops BEFORE routing figure-bearing
    # docs to Ovis, so the dots-ocr GPU stage never co-loads Ovis. The OvisFigureOCRTask
    # is then run in its OWN GPU container via ``run_ovis_app``. This mirrors upstream's
    # two-GPU split (dots and Ovis as separate @function tasks) for our single-process
    # wrapper. Default False = Ovis runs in-process right after dots (unchanged).
    defer_ovis: bool = False
    table_summarization: Optional[bool] = False
    table_summarization_prompt: Optional[str] = None
    table_merging: bool = False
    figure_summarization: Optional[bool] = False
    figure_summarization_prompt: Optional[str] = None
    figure_ocr_prompt: Optional[str] = None  # For automatic figure OCR in the `dots-ocr` path
    # This is to make the full page image in table and figure summarization optional
    chart_extraction: Optional[bool] = False
    table_cell_grounding: Optional[bool] = False
    figure_grounding: Optional[bool] = False
    key_value_extraction: Optional[bool] = False
    include_full_page_image: Optional[bool] = False
    ignore_sections: Optional[Set[PageFragmentType]] = None
    org_quota: Optional[OrganizationQuotaRequest] = None
    xpage_header_detection: bool = False
    include_images: Optional[bool] = False
    form_filling: Optional[FormFillingRequest] = None
