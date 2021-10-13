from typing import TypeVar, NewType

PlayerName = NewType("PlayerName", str)
""" Strings that represent player's names/IDs. """

X = TypeVar("X")
""" Generic variable for a player's state."""

U = TypeVar("U")
""" Generic variable for a player's commands. """

Y = TypeVar("Y")
""" Generic variable for the player's observations. """

RP = TypeVar("RP")
""" Generic variable for the Personal Reward. """

RJ = TypeVar("RJ")
""" Generic variable for the Joint Reward. """
