from bisect import bisect_left

import streamlit as st


# ==========================================
# Binary Search Tree Node
# ==========================================

class BSTNode:
    """
    Node used in Binary Search Tree
    """

    def __init__(self, key):
        self.key = key

        self.left = None

        self.right = None


# ==========================================
# Binary Search Tree
# ==========================================

class BinarySearchTree:
    """
    Binary Search Tree Implementation

    Used to store dictionary terms.
    """

    def __init__(self):

        self.root = None

    def insert(self, root, key):
        """
        Binary Search Tree Insertion
        :param root: Root node
        :param key: Key to insert into
        :return: Tree
        """

        if root is None:
            return BSTNode(key)

        if key < root.key:

            root.left = self.insert(
                root.left,
                key
            )

        elif key > root.key:

            root.right = self.insert(
                root.right,
                key
            )

        return root

    def search(self, root, key):
        """
        Binary Search Tree Search
        :param root: Root node
        :param key: Key to search
        :return: Node
        """

        if root is None:
            return False

        if root.key == key:
            return True

        if key < root.key:
            return self.search(root.left, key)

        return self.search(root.right, key)


# ==========================================
# B-Tree Node
# ==========================================

class BTreeNode:

    def __init__(self, leaf=False):
        self.leaf = leaf

        self.keys = []

        self.children = []


# ==========================================
# B-Tree
# ==========================================

class BTree:

    def __init__(self):
        self.root = BTreeNode(True)

    def insert(self, key):
        """
        Simplified B-Tree insertion

        Keys are maintained in sorted order.
        """

        self.root.keys.append(key)

        self.root.keys.sort()

    def search(self, key):
        """
        Binary search inside B-Tree node
        """

        idx = bisect_left(
            self.root.keys,
            key
        )

        return (
                idx < len(self.root.keys)
                and self.root.keys[idx] == key
        )


def build_balanced_bst(bst, terms):
    """
    Build balanced BST
    :param bst: BST
    :param terms: Terms
    :return: Balanced BST
    """
    if not terms:
        return

    mid = len(terms) // 2

    bst.root = bst.insert(
        bst.root,
        terms[mid]
    )

    build_balanced_bst(
        bst,
        terms[:mid]
    )

    build_balanced_bst(
        bst,
        terms[mid + 1:]
    )
