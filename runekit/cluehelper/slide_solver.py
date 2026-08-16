"""Solver for the 5x5 sliding puzzle.

Board is a list of 25 part indices in row-major grid order; part 24 is the
blank. Solved means board[cell] == cell. The solution is a list of grid cells
to click, each being the tile that slides into the blank.

Solves rows top-down until two remain, then columns left-to-right, finishing
with the last three tiles. Each subgoal is an exact BFS over the positions of
the tracked tiles and the blank, with all other tiles anonymous: small state
spaces, provably no deadlocks, and locally optimal move sequences.
"""
from collections import deque
from typing import List, Tuple

N = 5
BLANK = 24


def _rc(cell: int) -> Tuple[int, int]:
    return divmod(cell, N)


def _cell(r: int, c: int) -> int:
    return r * N + c


def _neighbors(cell: int):
    r, c = _rc(cell)
    if r > 0:
        yield cell - N
    if r < N - 1:
        yield cell + N
    if c > 0:
        yield cell - 1
    if c < N - 1:
        yield cell + 1


def is_solvable(board: List[int]) -> bool:
    # odd board width: solvable iff the tile permutation has even inversions
    perm = [p for p in board if p != BLANK]
    inversions = sum(
        1
        for i in range(len(perm))
        for j in range(i + 1, len(perm))
        if perm[i] > perm[j]
    )
    return inversions % 2 == 0


class _State:
    def __init__(self, board: List[int]):
        self.board = list(board)
        self.blank = self.board.index(BLANK)
        self.moves: List[int] = []
        self.locked = set()

    def click(self, cell: int):
        assert cell in list(_neighbors(self.blank)), (cell, self.blank)
        assert cell not in self.locked
        self.board[self.blank] = self.board[cell]
        self.board[cell] = BLANK
        self.blank = cell
        self.moves.append(cell)

    def solve_pieces(self, parts: List[int], dests: List[int]):
        """BFS the tracked parts to dests (anonymous other tiles), apply the
        move sequence, and lock the destination cells."""
        start = (tuple(self.board.index(p) for p in parts), self.blank)
        goal = tuple(dests)
        if start[0] == goal:
            self.locked.update(dests)
            return

        prev = {start: None}
        queue = deque([start])
        end = None
        while queue:
            state = queue.popleft()
            positions, blank = state
            for nb in _neighbors(blank):
                if nb in self.locked:
                    continue
                new_positions = tuple(blank if p == nb else p for p in positions)
                nxt = (new_positions, nb)
                if nxt in prev:
                    continue
                prev[nxt] = state
                if new_positions == goal:
                    end = nxt
                    queue.clear()
                    break
                queue.append(nxt)

        if end is None:
            raise ValueError(f"no solution for parts {parts} to {dests}")

        path = []
        node = end
        while prev[node] is not None:
            path.append(node[1])
            node = prev[node]
        for cell in reversed(path):
            self.click(cell)
        self.locked.update(dests)


def solve(board: List[int]) -> List[int]:
    if sorted(board) != list(range(N * N)):
        raise ValueError("board is not a permutation of 0..24")
    if not is_solvable(board):
        raise ValueError("board is not solvable")

    st = _State(board)

    # rows 0..2: three single placements, then the final two as a pair
    for r in range(N - 2):
        for c in range(N - 2):
            st.solve_pieces([_cell(r, c)], [_cell(r, c)])
        st.solve_pieces(
            [_cell(r, N - 2), _cell(r, N - 1)],
            [_cell(r, N - 2), _cell(r, N - 1)],
        )

    # columns 0..1 of the bottom two rows as vertical pairs
    for c in range(N - 3):
        st.solve_pieces(
            [_cell(N - 2, c), _cell(N - 1, c)],
            [_cell(N - 2, c), _cell(N - 1, c)],
        )

    # remaining 2x3 block holds five tiles and the blank; finish exactly
    tail = [c for c in range(N * N) if c not in st.locked and c != _cell(N - 1, N - 1)]
    st.solve_pieces(tail, tail)

    assert all(st.board[i] == i for i in range(N * N))
    return st.moves
