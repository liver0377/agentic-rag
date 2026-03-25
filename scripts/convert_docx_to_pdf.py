"""
将 data/pdfs 目录下的所有 Word 文档转换为 PDF，并移动到 data/docx 目录
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from docx2pdf import convert
except ImportError:
    print("正在安装 docx2pdf...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "docx2pdf"])
    from docx2pdf import convert


def main():
    base_dir = Path(r"C:\Users\liver0377\Documents\workspace\MODULAR-RAG-MCP-SERVER")
    pdfs_dir = base_dir / "data" / "pdfs"
    docx_dir = base_dir / "data" / "docx"

    docx_dir.mkdir(parents=True, exist_ok=True)

    docx_files = list(pdfs_dir.rglob("*.docx")) + list(pdfs_dir.rglob("*.doc"))

    if not docx_files:
        print("未找到 Word 文档")
        return

    print(f"找到 {len(docx_files)} 个 Word 文档")

    for docx_file in docx_files:
        try:
            relative_path = docx_file.relative_to(pdfs_dir)
            pdf_file = docx_file.with_suffix(".pdf")

            print(f"转换: {relative_path}")
            convert(str(docx_file), str(pdf_file))
            print(f"  -> PDF: {pdf_file.name}")

            dest_docx = docx_dir / relative_path
            dest_docx.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(docx_file), str(dest_docx))
            print(f"  -> 移动到: {dest_docx}")

        except Exception as e:
            print(f"处理 {docx_file.name} 失败: {e}")

    print("\n完成!")


if __name__ == "__main__":
    main()
