"""Small Newick parser for fast scoring tests and lightweight CLI use."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NewickNode:
    name: str | None = None
    children: list["NewickNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class NewickError(ValueError):
    """Raised when a Newick string cannot be parsed."""


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text.strip()
        self.i = 0

    def parse(self) -> NewickNode:
        if not self.text:
            raise NewickError("empty Newick string")
        node = self._subtree()
        self._skip_ws()
        if self._peek() == ";":
            self.i += 1
        self._skip_ws()
        if self.i != len(self.text):
            raise NewickError(f"unexpected trailing text at offset {self.i}")
        return node

    def _subtree(self) -> NewickNode:
        self._skip_ws()
        if self._peek() == "(":
            self.i += 1
            children = [self._subtree()]
            while True:
                self._skip_ws()
                char = self._peek()
                if char == ",":
                    self.i += 1
                    children.append(self._subtree())
                    continue
                if char == ")":
                    self.i += 1
                    break
                raise NewickError(f"expected ',' or ')' at offset {self.i}")
            name = self._label()
            self._branch_length()
            return NewickNode(name=name or None, children=children)

        name = self._label()
        if not name:
            raise NewickError(f"expected leaf label at offset {self.i}")
        self._branch_length()
        return NewickNode(name=name)

    def _label(self) -> str:
        self._skip_ws()
        if self._peek() in {":", ",", ")", "(", ";", ""}:
            return ""
        if self._peek() == "'":
            self.i += 1
            chars: list[str] = []
            while self.i < len(self.text):
                char = self.text[self.i]
                self.i += 1
                if char == "'":
                    if self.i < len(self.text) and self.text[self.i] == "'":
                        chars.append("'")
                        self.i += 1
                        continue
                    return "".join(chars)
                chars.append(char)
            raise NewickError("unterminated quoted label")

        start = self.i
        while self.i < len(self.text) and self.text[self.i] not in ":,();":
            self.i += 1
        return self.text[start : self.i].strip()

    def _branch_length(self) -> None:
        self._skip_ws()
        if self._peek() != ":":
            return
        self.i += 1
        while self.i < len(self.text) and self.text[self.i] not in ",);":
            self.i += 1

    def _peek(self) -> str:
        if self.i >= len(self.text):
            return ""
        return self.text[self.i]

    def _skip_ws(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1


def parse_newick(text: str) -> NewickNode:
    """Parse a Newick string into a lightweight tree."""

    return _Parser(text).parse()


def leaf_names(root: NewickNode) -> list[str]:
    names: list[str] = []

    def visit(node: NewickNode) -> None:
        if node.is_leaf:
            if node.name is None:
                raise NewickError("leaf without a label")
            names.append(node.name)
            return
        for child in node.children:
            visit(child)

    visit(root)
    return sorted(names)

