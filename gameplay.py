import streamlit as st

from utils.board_utils import relative_to_absolute
from ultimatetictactoe import ultimatetictactoe

SYMBOLS = {1: "X", 2: "O"}

def init_state():
	if "env" not in st.session_state:
		st.session_state.env = ultimatetictactoe.env(render_mode=None)
		st.session_state.env.reset()
		st.current_player = st.session_state.env.agent_selection

def render():
	for r in range(3):
		super_cols = st.columns([1, 0.05, 1, 0.05, 1])
		col_idcs = [0, 2, 4]

		for c in range(3):
			board_idx = r * 3 + c
			with super_cols[col_idcs[c]]:
				with st.container(border=True):
					render_sub_board(board_idx)

def render_sub_board(board_idx):
	env = st.session_state.env
	is_active = env.board.current_pos == board_idx or env.board.current_pos == -1
	winner = env.board.super_cells[board_idx]

	if winner:
		# st.markdown(f"**{SYMBOLS.get(winner)} wins this board**")
		st.button(SYMBOLS.get(winner), disabled=True, use_container_width=True)
		return
		
	for row in range(3):
		cols = st.columns([1, 1, 1])
		for col in range(3):
			cell_idx = row * 3 + col
			abs_idx = relative_to_absolute(board_idx, cell_idx)
			cell_value = SYMBOLS.get(env.board.cells[abs_idx])
			with cols[col]:
				if cell_value:
					st.button(
						cell_value,
						key=f"cell_{board_idx}_{cell_idx}",
						disabled=True,
					)
				elif not is_active or winner:
					st.button(
						" • ",
						key=f"cell_{board_idx}_{cell_idx}",
						disabled=True,
					)
				else:
					if st.button("[  ]", key=f"cell_{board_idx}_{cell_idx}"):
						make_move(abs_idx)

def make_move(action):
	env = st.session_state.env
	print(f"playing {action}")
	env.step(action)
	st.rerun()

init_state()
st.title("SuperTris")
render()