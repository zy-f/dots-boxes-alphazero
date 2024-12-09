# AlphaZero for Dots and Boxes
Final project for CS238 in Fall 2024 by Renee, Frank, and Christos.

An implementation of AlphaZero for the game of Dots and Boxes.

## Setup
The `main` branch is designed to make it easy to play with our final models.
For model training purposes, please switch to the `dev` branch.

To play with our model:
- Setup virtual environemt (recommended). For example:
    - `python -m venv dnb_env`
    - `source dnb_env/bin/activate`
- Install requirements: `pip install -r requirements.txt`

Instructions for dots and boxes:
- On your turn, draw a line between two adjacent dots (vertical and horizontal only).
- If you complete a box, you get one point and get another turn.
- Most boxes after the board is full wins.

## Commands
### `main` branch
To play with our final trained models:
```
python play.py
```
- Specify yourself as `human` and your opponent as `alphazero` (Player 1 goes first)
    - Note: you can also choose to play against other baselines or pit AlphaZero against them.
        - `greedy` views each move in 3 tiers: good (immediately completing a box), bad (sets opponent up to complete a box), and neutral (neither). It randomly picks a move in the best available tier.
        - `random` randomly picks any legal move.
- Pick your board size from 2 to 4.
- Instructions on how to use our UI will be shown.
- Good luck!

(If you want to win, we highly recommend watching the [Numberphile video](https://www.youtube.com/watch?v=KboGyIilP6k) that inspired our project.)

### `dev` branch
To train a model with specified config file:
```
python train_alphazero.py <path-to-config>
```

To play with the trained model:
```
python play.py <path-to-config>
```
