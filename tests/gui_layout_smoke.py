"""Smoke-check the reworked queue layout: widget heights + how many rows fit."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QScrollArea, QTableWidgetItem
from PySide6.QtCore import Qt

from src.gui import AsrWindow, COL_FILE, COL_STATUS


def fake_rows(window, count: int) -> None:
    window.file_table.setRowCount(count)
    for row in range(count):
        item = window.file_table.item(row, COL_FILE)
        if item is None:
            from PySide6.QtWidgets import QTableWidgetItem
            item = QTableWidgetItem(f"sample-clip-{row:03d}.mp4")
            window.file_table.setItem(row, COL_FILE, item)
        status = window.file_table.item(row, COL_STATUS)
        if status is None:
            from PySide6.QtWidgets import QTableWidgetItem
            status = QTableWidgetItem("等待中")
            window.file_table.setItem(row, COL_STATUS, status)
    window.paths = [None] * count  # only used to flip visibility of the drop card


def main() -> int:
    app = QApplication([])
    window = AsrWindow()
    window.resize(1180, 860)
    window.show()
    app.processEvents()

    fake_rows(window, 20)
    window._refresh_queue_view()
    app.processEvents()
    app.processEvents()

    row_height = window.file_table.verticalHeader().defaultSectionSize()
    viewport = window.file_table.viewport().height()
    header_height = window.file_table.horizontalHeader().height()
    visible_rows = max(0, (viewport - header_height) // row_height)

    print("=== widget heights (window 1180x860) ===")
    for name, widget in (
        ("header card", window.centralWidget().layout().itemAt(0).widget()),
        ("drop card visible", window.drop_card.isVisible()),
        ("list card", window.file_table.parentWidget().parentWidget().parentWidget()),
        ("table", window.file_table),
        ("log card", window.log_card),
        ("gpu card", window.gpu_util_bar.parentWidget()),
    ):
        if isinstance(name, str) and isinstance(widget, bool):
            print(f"{name:22s}{widget}")
        else:
            print(f"{name:22s}{widget.height()}px")

    print("=== queue capacity ===")
    print(f"row height               {row_height}px")
    print(f"table viewport           {viewport}px")
    print(f"visible data rows        {visible_rows}")
    print(f"drop card visible        {window.drop_card.isVisible()}")
    print(f"columns                  {[window.file_table.columnWidth(c) for c in range(4)]}")

    left_panel = window.drop_card.parentWidget()
    settings_scroll = window.findChild(QScrollArea)
    gpu_card = window.gpu_util_bar.parentWidget()
    print("=== right column fill ===")
    print(f"settings viewport        {settings_scroll.viewport().height()}px")
    print(f"settings content         {settings_scroll.widget().height()}px")
    print(f"gpu card height          {gpu_card.height()}px  (stretch fills the gap)")
    print(f"left column bottom       {left_panel.geometry().bottom()}")

    # Empty state: the drop card comes back and the queue shows the placeholder.
    window.paths = []
    window.file_table.setRowCount(0)
    window._refresh_queue_view()
    app.processEvents()
    print("=== empty state ===")
    print(f"drop card visible        {window.drop_card.isVisible()}  height={window.drop_card.height()}px")
    print(f"queue stack index        {window.queue_stack.currentIndex()}  (0 = placeholder)")

    print("=== intake & removal ===")
    root = Path(__file__).resolve().parents[1]
    window._add_paths_from_strings([str(root / "output")])
    print(f"rows from directory      {window.file_table.rowCount()}")
    print(f"progress widgets tracked {len(window._progress_bars)}")
    window._add_paths_from_strings([str(root / "demo.m4a")])
    print(f"rows after single file   {window.file_table.rowCount()}")
    window._add_paths_from_strings([str(root / "demo.m4a")])  # duplicate -> no-op
    print(f"rows after duplicate     {window.file_table.rowCount()}")
    if window.file_table.rowCount():
        window.file_table.selectRow(0)
        window._remove_selected()
    print(f"rows after remove        {window.file_table.rowCount()}")
    print(f"progress widgets after   {len(window._progress_bars)}")
    window._clear_files()
    print(f"rows after clear         {window.file_table.rowCount()}  bar cache={len(window._progress_bars)}")

    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
