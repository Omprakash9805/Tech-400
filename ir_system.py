"""
ir_system.py - A minimal Information Retrieval system.

Components
----------
1. Document store      : add documents from a file or from typed text
2. Linguistic module   : tokenization, normalization, stop-word removal
3. Dictionary          : unique terms + document frequency
4. Inverted index      : term -> sorted posting list of docIDs
5. Boolean retrieval   : AND, OR, NOT with parentheses

Run `python3 ir_system.py` for the interactive menu.
"""

import json
import os
import re
import sys

# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "do", "does", "each", "either", "every", "for", "from", "has", "have",
    "how", "in", "into", "is", "it", "its", "many", "may", "must", "no",
    "not", "of", "on", "or", "so", "still", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to", "two",
    "up", "used", "was", "which", "while", "who", "why", "with", "within",
}

# NOT / AND / OR are query operators, so they must never be stripped from a
# query even though "and", "not" and "or" are stop words in a document.
OPERATORS = {"AND", "OR", "NOT"}
PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}

INDEX_FILE = "index.json"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


# --------------------------------------------------------------------------
# 2. LINGUISTIC MODULE
# --------------------------------------------------------------------------

def tokenize(text):
    """Lowercase the text and cut it into alphanumeric tokens.

    Hyphenated forms such as 'tf-idf' are kept as one token because splitting
    them would destroy the meaning of the term.
    """
    return TOKEN_PATTERN.findall(text.lower())


def normalize(tokens, remove_stopwords=True):
    """Drop stop words and any token of length 1."""
    out = []
    for t in tokens:
        if len(t) < 2:
            continue
        if remove_stopwords and t in STOP_WORDS:
            continue
        out.append(t)
    return out


def preprocess(text):
    return normalize(tokenize(text))


# --------------------------------------------------------------------------
# 3. POSTING LIST ALGORITHMS (linear merges over sorted lists)
# --------------------------------------------------------------------------

def intersect(p1, p2):
    """AND - walk both sorted lists once. O(len(p1) + len(p2))."""
    result, i, j = [], 0, 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            result.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return result


def union(p1, p2):
    """OR - merge of two sorted lists, duplicates collapsed."""
    result, i, j = [], 0, 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            result.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            result.append(p1[i])
            i += 1
        else:
            result.append(p2[j])
            j += 1
    result.extend(p1[i:])
    result.extend(p2[j:])
    return result


def difference(p1, p2):
    """AND NOT - every docID in p1 that is absent from p2."""
    result, i, j = [], 0, 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            result.append(p1[i])
            i += 1
        else:
            j += 1
    result.extend(p1[i:])
    return result


# --------------------------------------------------------------------------
# 4. THE RETRIEVAL SYSTEM
# --------------------------------------------------------------------------

class IRSystem:

    def __init__(self):
        self.documents = {}       # docID -> {'title', 'path', 'length'}
        self.index = {}           # term  -> sorted list of docIDs
        self.next_id = 1

    # ---------------- document handling ----------------

    def add_document(self, text, title, path="(typed in)"):
        """Index one document and return its docID."""
        doc_id = self.next_id
        self.next_id += 1

        terms = preprocess(text)
        self.documents[doc_id] = {
            "title": title,
            "path": path,
            "length": len(terms),
        }

        for term in set(terms):                 # set() -> each docID once
            postings = self.index.setdefault(term, [])
            postings.append(doc_id)
            postings.sort()                     # keep postings in docID order

        return doc_id

    def add_from_file(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        title = text.strip().split("\n")[0][:60]
        return self.add_document(text, title, path)

    @staticmethod
    def _natural_key(name):
        """Sort doc2 before doc10 by comparing digit runs as numbers."""
        return [int(part) if part.isdigit() else part
                for part in re.split(r"(\d+)", name)]

    def add_folder(self, folder):
        added = []
        for name in sorted(os.listdir(folder), key=self._natural_key):
            if name.endswith(".txt"):
                added.append(self.add_from_file(os.path.join(folder, name)))
        return added

    # ---------------- dictionary ----------------

    def dictionary(self):
        """Sorted list of (term, document frequency)."""
        return sorted((t, len(p)) for t, p in self.index.items())

    def postings(self, term):
        return self.index.get(term.lower(), [])

    # ---------------- Boolean query engine ----------------

    @staticmethod
    def parse_query(query):
        """Split a query into tokens, keeping operators and parentheses."""
        raw = query.replace("(", " ( ").replace(")", " ) ").split()
        out = []
        for tok in raw:
            if tok.upper() in OPERATORS or tok in "()":
                out.append(tok.upper() if tok not in "()" else tok)
            else:
                clean = tokenize(tok)
                out.extend(clean)
        return out

    @staticmethod
    def to_postfix(tokens):
        """Shunting-yard: infix query -> postfix (Reverse Polish) form.

        Postfix removes the need to worry about precedence during evaluation,
        so 'a OR b AND c' correctly binds AND before OR.
        """
        output, stack = [], []
        for tok in tokens:
            if tok == "(":
                stack.append(tok)
            elif tok == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("unbalanced parentheses in query")
                stack.pop()
            elif tok in OPERATORS:
                while (stack and stack[-1] in OPERATORS
                       and PRECEDENCE[stack[-1]] >= PRECEDENCE[tok]):
                    output.append(stack.pop())
                stack.append(tok)
            else:
                output.append(tok)
        while stack:
            top = stack.pop()
            if top == "(":
                raise ValueError("unbalanced parentheses in query")
            output.append(top)
        return output

    def all_docs(self):
        return sorted(self.documents.keys())

    def evaluate(self, query):
        """Evaluate a Boolean query and return a sorted list of docIDs."""
        tokens = self.parse_query(query)
        if not tokens:
            return []
        postfix = self.to_postfix(tokens)
        stack = []

        for tok in postfix:
            if tok == "NOT":
                if not stack:
                    raise ValueError("NOT is missing an operand")
                operand = stack.pop()
                stack.append(difference(self.all_docs(), operand))
            elif tok in ("AND", "OR"):
                if len(stack) < 2:
                    raise ValueError(f"{tok} is missing an operand")
                right, left = stack.pop(), stack.pop()
                stack.append(intersect(left, right) if tok == "AND"
                             else union(left, right))
            else:
                stack.append(list(self.postings(tok)))

        if len(stack) != 1:
            raise ValueError("malformed query")
        return stack[0]

    # ---------------- persistence ----------------

    def save(self, path=INDEX_FILE):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"documents": self.documents,
                       "index": self.index,
                       "next_id": self.next_id}, fh, indent=1)

    def load(self, path=INDEX_FILE):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.documents = {int(k): v for k, v in data["documents"].items()}
        self.index = data["index"]
        self.next_id = data["next_id"]


# --------------------------------------------------------------------------
# 5. DISPLAY HELPERS
# --------------------------------------------------------------------------

def show_documents(ir):
    print(f"\n{'DocID':<7}{'Title':<42}{'Terms':>6}  Source")
    print("-" * 78)
    for did in ir.all_docs():
        d = ir.documents[did]
        print(f"{did:<7}{d['title'][:40]:<42}{d['length']:>6}  {d['path']}")
    print("-" * 78)
    print(f"{len(ir.documents)} documents indexed.\n")


def show_dictionary(ir, limit=None):
    entries = ir.dictionary()
    print(f"\nDICTIONARY - {len(entries)} unique terms")
    print(f"{'Term':<20}{'df':>4}")
    print("-" * 24)
    for term, df in entries[:limit]:
        print(f"{term:<20}{df:>4}")
    if limit and len(entries) > limit:
        print(f"... {len(entries) - limit} more terms")
    print()


def show_index(ir, limit=None):
    entries = ir.dictionary()
    print(f"\nINVERTED INDEX - {len(entries)} terms")
    print(f"{'Term':<18}{'df':>3}   Postings (docIDs)")
    print("-" * 58)
    for term, df in entries[:limit]:
        plist = ", ".join(str(d) for d in ir.postings(term))
        print(f"{term:<18}{df:>3}   [{plist}]")
    if limit and len(entries) > limit:
        print(f"... {len(entries) - limit} more terms")
    print()


def show_results(ir, query, docs):
    print(f"\nQuery : {query}")
    print(f"Tokens: {ir.parse_query(query)}")
    print(f"Postfix: {ir.to_postfix(ir.parse_query(query))}")
    if not docs:
        print("Result: no matching documents.\n")
        return
    print(f"Result: {len(docs)} document(s) -> {docs}")
    for did in docs:
        print(f"   [{did}] {ir.documents[did]['title']}")
    print()


# --------------------------------------------------------------------------
# 6. INTERACTIVE MENU
# --------------------------------------------------------------------------

MENU = """
=========== BOOLEAN INFORMATION RETRIEVAL SYSTEM ===========
 1. Add document from a file
 2. Add document by typing text
 3. Load every .txt file from a folder
 4. Show document collection
 5. Show dictionary
 6. Show inverted index
 7. Run a Boolean query
 8. Save index to disk
 9. Load index from disk
 0. Exit
============================================================"""


def main():
    ir = IRSystem()
    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()

        try:
            if choice == "1":
                path = input("File path: ").strip()
                did = ir.add_from_file(path)
                print(f"Added as docID {did}.")
            elif choice == "2":
                title = input("Title: ").strip()
                print("Enter text, finish with a blank line:")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                did = ir.add_document(" ".join(lines), title)
                print(f"Added as docID {did}.")
            elif choice == "3":
                folder = input("Folder path: ").strip()
                ids = ir.add_folder(folder)
                print(f"Added {len(ids)} documents: {ids}")
            elif choice == "4":
                show_documents(ir)
            elif choice == "5":
                show_dictionary(ir, limit=25)
            elif choice == "6":
                show_index(ir, limit=25)
            elif choice == "7":
                q = input("Boolean query: ").strip()
                show_results(ir, q, ir.evaluate(q))
            elif choice == "8":
                ir.save()
                print(f"Index written to {INDEX_FILE}.")
            elif choice == "9":
                ir.load()
                print(f"Index loaded ({len(ir.documents)} documents).")
            elif choice == "0":
                print("Goodbye.")
                break
            else:
                print("Invalid option.")
        except FileNotFoundError as e:
            print(f"Error: file not found - {e.filename}")
        except ValueError as e:
            print(f"Query error: {e}")


if __name__ == "__main__":
    main()
