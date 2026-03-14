import torch
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from rag.config import settings


pipeline_options = PdfPipelineOptions()

pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

filepath = settings.RAW_DATA_DIR / "test.pdf"
filepath_w = settings.RAW_DATA_DIR / "t.md"

result = converter.convert(filepath)

markdown_text = result.document.export_to_markdown()

with open(filepath_w, "w+", encoding="UTF-8") as f:
    f.write(markdown_text)
