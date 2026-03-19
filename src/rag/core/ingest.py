from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_file(filepath: Path) -> str:
    pipeline_options = PdfPipelineOptions()

    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CUDA
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(filepath)

    markdown_text = result.document.export_to_markdown()

    return markdown_text


def text_splitter(docs_text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_text(docs_text)

    return chunks
