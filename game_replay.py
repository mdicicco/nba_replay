#!/usr/bin/env python3
"""
NBA Game Replay Visualization
Loads player tracking data and play-by-play events to create an animated 2D replay
of an NBA game using pygame with event-based navigation and adjustable speed.
Includes vector analysis of player positions relative to ball carrier and hoop.
"""

import json
import math
import os
import pygame
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque

from event_info import event_dict, event_desc, shot_desc

# ============================================================================
# Constants
# ============================================================================

# Data directory for JSON files
DATA_DIR = "data"

# NBA Court dimensions in feet (tracking data uses feet)
COURT_LENGTH = 94.0  # feet
COURT_WIDTH = 50.0   # feet

# Window dimensions
COURT_SCALE = 12  # pixels per foot
COURT_WIDTH_PX = int(COURT_LENGTH * COURT_SCALE)
COURT_HEIGHT_PX = int(COURT_WIDTH * COURT_SCALE)
ANALYSIS_PANEL_WIDTH = 420  # Right side analysis panel
WINDOW_WIDTH = COURT_WIDTH_PX + ANALYSIS_PANEL_WIDTH
WINDOW_HEIGHT = COURT_HEIGHT_PX + 250  # Extra space for controls

# Hoop positions in feet (from baseline, centered)
LEFT_HOOP = (5.25, COURT_WIDTH / 2)   # 5.25 feet from left baseline
RIGHT_HOOP = (COURT_LENGTH - 5.25, COURT_WIDTH / 2)  # 5.25 feet from right baseline

# Colors - Retro basketball court aesthetic
HARDWOOD_DARK = (185, 140, 95)
HARDWOOD_LIGHT = (210, 175, 130)
COURT_LINE_COLOR = (255, 255, 255)
THREE_POINT_COLOR = (200, 50, 50)
KEY_COLOR = (180, 120, 80)
CENTER_CIRCLE_COLOR = (200, 50, 50)

# Team colors (will be set dynamically)
HOME_COLOR = (225, 0, 40)      # Default red
AWAY_COLOR = (0, 80, 180)      # Default blue
BALL_COLOR = (255, 140, 0)     # Orange basketball
BALL_SHADOW = (200, 100, 0)

# UI Colors
BG_COLOR = (15, 15, 25)
TEXT_COLOR = (240, 240, 240)
SCORE_COLOR = (255, 220, 100)
EVENT_BG_COLOR = (25, 25, 40)
HEADER_COLOR = (45, 45, 65)
BUTTON_COLOR = (60, 120, 200)
BUTTON_HOVER_COLOR = (80, 150, 230)
BUTTON_TEXT_COLOR = (255, 255, 255)
SLIDER_BG_COLOR = (60, 60, 80)
SLIDER_TRACK_COLOR = (100, 100, 120)
SLIDER_HANDLE_COLOR = (255, 200, 80)

# Analysis panel colors
ANALYSIS_BG_COLOR = (20, 25, 35)
VECTOR_BALL_TO_HOOP = (255, 200, 50)      # Yellow - ball to hoop
VECTOR_CARRIER_TO_HOOP = (50, 255, 100)   # Green - carrier to hoop
VECTOR_OFFENSE_TO_CARRIER = (100, 200, 255)  # Light blue - offense to carrier
VECTOR_DEFENSE_TO_CARRIER = (255, 100, 100)  # Red - defense to carrier
# Note: Centroid colors now use team colors dynamically

# Pass chain colors
PASS_CHAIN_COLOR = (255, 220, 100)        # Gold for pass chain display

# Player circle size
PLAYER_RADIUS = 14
BALL_RADIUS = 8

# Speed options for slider
SPEED_OPTIONS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_SPEED_INDEX = 3  # 1.0x

# Frames per second target
TARGET_FPS = 60


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Player:
    player_id: int
    first_name: str
    last_name: str
    jersey: str
    position: str
    team_id: int
    team_abbr: str
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
    
    @property
    def display_name(self) -> str:
        return f"{self.first_name[0]}. {self.last_name}"


@dataclass
class Moment:
    quarter: int
    timestamp: int
    game_clock: float  # seconds remaining in quarter
    shot_clock: Optional[float]
    ball_pos: Tuple[float, float, float]  # x, y, z
    player_positions: Dict[int, Tuple[float, float]]  # player_id -> (x, y)


@dataclass 
class PlayByPlayEvent:
    event_num: int
    event_type: int
    action_type: int
    period: int
    game_clock_str: str
    home_desc: Optional[str]
    neutral_desc: Optional[str]
    visitor_desc: Optional[str]
    score: Optional[str]
    player1_id: Optional[int]
    player1_name: Optional[str]
    player1_team_abbr: Optional[str]
    player2_id: Optional[int]
    player2_name: Optional[str]
    frame_index: int = -1  # Will be set during loading


@dataclass
class GameData:
    game_id: str
    game_date: str
    home_team_name: str
    home_team_abbr: str
    home_team_id: int
    away_team_name: str
    away_team_abbr: str
    away_team_id: int
    players: Dict[int, Player]
    moments: List[Moment]
    events: List[PlayByPlayEvent]


@dataclass
class Vector2D:
    """2D vector with magnitude and angle."""
    dx: float
    dy: float
    
    @property
    def magnitude(self) -> float:
        return math.sqrt(self.dx ** 2 + self.dy ** 2)
    
    @property
    def angle_degrees(self) -> float:
        """Angle in degrees from positive x-axis."""
        return math.degrees(math.atan2(self.dy, self.dx))
    
    def normalized(self) -> 'Vector2D':
        mag = self.magnitude
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.dx / mag, self.dy / mag)


@dataclass
class PlayerVector:
    """Vector from one point to a player with metadata."""
    player_id: int
    player_name: str
    jersey: str
    team_id: int
    vector: Vector2D
    distance: float
    angle: float


@dataclass
class PassEvent:
    """Records a pass from one player to another."""
    passer_id: int
    passer_name: str
    passer_jersey: str
    passer_pos: Tuple[float, float]
    receiver_id: int
    receiver_name: str
    receiver_jersey: str
    receiver_pos: Tuple[float, float]
    timestamp: int
    
    @property
    def pass_distance(self) -> float:
        return math.sqrt(
            (self.receiver_pos[0] - self.passer_pos[0]) ** 2 +
            (self.receiver_pos[1] - self.passer_pos[1]) ** 2
        )


@dataclass
class PassChain:
    """Chain of passes leading up to a shot."""
    passes: List[PassEvent] = field(default_factory=list)
    shooter_id: Optional[int] = None
    shooter_name: str = ""
    shooter_pos: Tuple[float, float] = (0, 0)
    shot_made: bool = False
    
    def add_pass(self, pass_event: PassEvent):
        self.passes.append(pass_event)
    
    def clear(self):
        self.passes = []
        self.shooter_id = None
        self.shooter_name = ""
        self.shooter_pos = (0, 0)
        self.shot_made = False
    
    @property
    def total_passes(self) -> int:
        return len(self.passes)
    
    @property
    def total_distance(self) -> float:
        return sum(p.pass_distance for p in self.passes)


@dataclass
class PositionAnalysis:
    """Analysis of player positions at a moment."""
    ball_carrier_id: Optional[int] = None
    ball_carrier_name: str = ""
    ball_carrier_pos: Tuple[float, float] = (0, 0)
    offensive_team_id: Optional[int] = None
    target_hoop: Tuple[float, float] = (0, 0)
    
    # Vectors
    ball_to_hoop: Optional[Vector2D] = None
    carrier_to_hoop: Optional[Vector2D] = None
    offense_to_carrier: List[PlayerVector] = field(default_factory=list)
    defense_to_carrier: List[PlayerVector] = field(default_factory=list)
    
    # Centroids
    offensive_centroid: Tuple[float, float] = (0, 0)
    defensive_centroid: Tuple[float, float] = (0, 0)


# ============================================================================
# Analysis Functions
# ============================================================================

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def calculate_vector(from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> Vector2D:
    """Calculate vector from one point to another."""
    return Vector2D(to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])


def find_ball_carrier(
    ball_pos: Tuple[float, float, float],
    player_positions: Dict[int, Tuple[float, float]],
    players: Dict[int, Player]
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the player closest to the ball (ball carrier) and their team.
    Returns (player_id, team_id) or (None, None) if no players.
    """
    if not player_positions:
        return None, None
    
    ball_xy = (ball_pos[0], ball_pos[1])
    min_dist = float('inf')
    carrier_id = None
    
    for pid, pos in player_positions.items():
        dist = calculate_distance(ball_xy, pos)
        if dist < min_dist:
            min_dist = dist
            carrier_id = pid
    
    if carrier_id and carrier_id in players:
        return carrier_id, players[carrier_id].team_id
    
    return None, None


def determine_target_hoop(
    offensive_team_id: int, 
    home_team_id: int, 
    quarter: int
) -> Tuple[float, float]:
    """
    Determine which hoop the offense is attacking based on team and quarter.
    
    Convention:
    - Home team attacks RIGHT hoop in Q1/Q2 (first half)
    - Home team attacks LEFT hoop in Q3/Q4 (second half)
    - Away team always attacks the opposite hoop
    - Teams switch sides at halftime
    """
    first_half = quarter <= 2
    
    # Home team's target hoop
    if offensive_team_id == home_team_id:
        # Home team attacks right in first half, left in second half
        return RIGHT_HOOP if first_half else LEFT_HOOP
    else:
        # Away team attacks the opposite
        return LEFT_HOOP if first_half else RIGHT_HOOP


def get_hoop_for_team(team_id: int, home_team_id: int, quarter: int) -> Tuple[float, float]:
    """Get which hoop a specific team is attacking."""
    return determine_target_hoop(team_id, home_team_id, quarter)


def analyze_positions(
    moment: Moment,
    players: Dict[int, Player],
    home_team_id: int,
    away_team_id: int
) -> PositionAnalysis:
    """Perform full position analysis for a moment."""
    analysis = PositionAnalysis()
    
    ball_pos = moment.ball_pos
    ball_xy = (ball_pos[0], ball_pos[1])
    
    # Find ball carrier
    carrier_id, offensive_team_id = find_ball_carrier(ball_pos, moment.player_positions, players)
    
    if not carrier_id:
        return analysis
    
    analysis.ball_carrier_id = carrier_id
    analysis.ball_carrier_name = players[carrier_id].display_name if carrier_id in players else ""
    analysis.ball_carrier_pos = moment.player_positions.get(carrier_id, (0, 0))
    analysis.offensive_team_id = offensive_team_id
    
    # Determine target hoop based on which team has possession
    analysis.target_hoop = determine_target_hoop(offensive_team_id, home_team_id, moment.quarter)
    
    # Calculate ball to hoop vector
    analysis.ball_to_hoop = calculate_vector(ball_xy, analysis.target_hoop)
    
    # Calculate carrier to hoop vector
    analysis.carrier_to_hoop = calculate_vector(analysis.ball_carrier_pos, analysis.target_hoop)
    
    # Separate players by team and calculate vectors
    offensive_positions = []
    defensive_positions = []
    
    for pid, pos in moment.player_positions.items():
        if pid == carrier_id:
            continue  # Skip the ball carrier
        
        if pid not in players:
            continue
        
        player = players[pid]
        vec = calculate_vector(analysis.ball_carrier_pos, pos)
        dist = calculate_distance(analysis.ball_carrier_pos, pos)
        
        player_vec = PlayerVector(
            player_id=pid,
            player_name=player.display_name,
            jersey=player.jersey,
            team_id=player.team_id,
            vector=vec,
            distance=dist,
            angle=vec.angle_degrees
        )
        
        if player.team_id == offensive_team_id:
            analysis.offense_to_carrier.append(player_vec)
            offensive_positions.append(pos)
        else:
            analysis.defense_to_carrier.append(player_vec)
            defensive_positions.append(pos)
    
    # Add carrier to offensive positions for centroid
    offensive_positions.append(analysis.ball_carrier_pos)
    
    # Calculate centroids
    if offensive_positions:
        ox = sum(p[0] for p in offensive_positions) / len(offensive_positions)
        oy = sum(p[1] for p in offensive_positions) / len(offensive_positions)
        analysis.offensive_centroid = (ox, oy)
    
    if defensive_positions:
        dx = sum(p[0] for p in defensive_positions) / len(defensive_positions)
        dy = sum(p[1] for p in defensive_positions) / len(defensive_positions)
        analysis.defensive_centroid = (dx, dy)
    
    # Sort by distance
    analysis.offense_to_carrier.sort(key=lambda x: x.distance)
    analysis.defense_to_carrier.sort(key=lambda x: x.distance)
    
    return analysis


# ============================================================================
# UI Components
# ============================================================================

class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.hovered = False
        self.font = pygame.font.Font(None, 28)
    
    def draw(self, screen: pygame.Surface):
        color = BUTTON_HOVER_COLOR if self.hovered else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2, border_radius=8)
        
        text_surface = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class SpeedSlider:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.speed_index = DEFAULT_SPEED_INDEX
        self.dragging = False
        self.font = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 28)
        
        # Calculate notch positions
        self.notch_positions = []
        for i in range(len(SPEED_OPTIONS)):
            notch_x = self.x + int((i / (len(SPEED_OPTIONS) - 1)) * self.width)
            self.notch_positions.append(notch_x)
    
    @property
    def speed(self) -> float:
        return SPEED_OPTIONS[self.speed_index]
    
    def draw(self, screen: pygame.Surface):
        # Draw track background
        track_rect = pygame.Rect(self.x, self.y + self.height // 2 - 4, self.width, 8)
        pygame.draw.rect(screen, SLIDER_TRACK_COLOR, track_rect, border_radius=4)
        
        # Draw notches and labels
        for i, notch_x in enumerate(self.notch_positions):
            # Notch line
            pygame.draw.line(screen, (150, 150, 150), 
                           (notch_x, self.y + self.height // 2 - 10),
                           (notch_x, self.y + self.height // 2 + 10), 2)
            
            # Label
            speed_val = SPEED_OPTIONS[i]
            if speed_val >= 1.0:
                label = f"{int(speed_val)}x" if speed_val == int(speed_val) else f"{speed_val}x"
            else:
                label = f"{speed_val}"
            
            text = self.font.render(label, True, (180, 180, 180))
            text_rect = text.get_rect(center=(notch_x, self.y + self.height - 5))
            screen.blit(text, text_rect)
        
        # Draw handle
        handle_x = self.notch_positions[self.speed_index]
        handle_rect = pygame.Rect(handle_x - 8, self.y + self.height // 2 - 12, 16, 24)
        pygame.draw.rect(screen, SLIDER_HANDLE_COLOR, handle_rect, border_radius=4)
        pygame.draw.rect(screen, (255, 255, 255), handle_rect, 2, border_radius=4)
        
        # Draw current speed label - positioned to the right, above the slider
        speed_text = f"Speed: {self.speed}x"
        speed_surface = self.font_large.render(speed_text, True, SCORE_COLOR)
        screen.blit(speed_surface, (self.x + self.width + 30, self.y - 5))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if clicking on or near the slider
            if (self.x - 20 <= event.pos[0] <= self.x + self.width + 20 and
                self.y <= event.pos[1] <= self.y + self.height):
                self.dragging = True
                self._update_from_mouse(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self._update_from_mouse(event.pos[0])
                return True
        return False
    
    def _update_from_mouse(self, mouse_x: int):
        # Snap to nearest notch
        min_dist = float('inf')
        nearest_index = self.speed_index
        for i, notch_x in enumerate(self.notch_positions):
            dist = abs(mouse_x - notch_x)
            if dist < min_dist:
                min_dist = dist
                nearest_index = i
        self.speed_index = nearest_index


# ============================================================================
# Data Loading Functions
# ============================================================================

def parse_game_clock(clock_str: str) -> float:
    """Convert MM:SS clock string to seconds."""
    parts = clock_str.split(':')
    return float(parts[0]) * 60 + float(parts[1])


def load_play_by_play(game_id: str) -> List[PlayByPlayEvent]:
    """Load and parse the play-by-play data."""
    with open(os.path.join(DATA_DIR, f'{game_id}_pbp.json'), 'r') as f:
        data = json.load(f)
    
    rows = data['resultSets'][0]['rowSet']
    events = []
    
    for row in rows:
        events.append(PlayByPlayEvent(
            event_num=row[1],
            event_type=row[2],
            action_type=row[3],
            period=row[4],
            game_clock_str=row[6],
            home_desc=row[7],
            neutral_desc=row[8],
            visitor_desc=row[9],
            score=row[10],
            player1_id=row[13],
            player1_name=row[14],
            player1_team_abbr=row[18],
            player2_id=row[20],
            player2_name=row[21]
        ))
    
    return events


def load_game(game_id: str) -> GameData:
    """Load all game data."""
    print(f"Loading game {game_id}...")
    
    # Load tracking data
    with open(os.path.join(DATA_DIR, f'{game_id}_svu.json'), 'r') as f:
        svu_data = json.load(f)
    
    game_date = svu_data['gamedate']
    
    # Parse team and player info
    first_event = svu_data['events'][0]
    home_info = first_event['home']
    away_info = first_event['visitor']
    
    players = {}
    
    for p in home_info['players']:
        players[p['playerid']] = Player(
            player_id=p['playerid'],
            first_name=p['firstname'],
            last_name=p['lastname'],
            jersey=p['jersey'],
            position=p['position'],
            team_id=home_info['teamid'],
            team_abbr=home_info['abbreviation']
        )
    
    for p in away_info['players']:
        players[p['playerid']] = Player(
            player_id=p['playerid'],
            first_name=p['firstname'],
            last_name=p['lastname'],
            jersey=p['jersey'],
            position=p['position'],
            team_id=away_info['teamid'],
            team_abbr=away_info['abbreviation']
        )
    
    # Parse moments
    moments = []
    for event in svu_data['events']:
        for m in event['moments']:
            quarter = m[0]
            timestamp = m[1]
            game_clock = m[2]
            shot_clock = m[3]
            
            positions = m[5]
            ball_pos = (positions[0][2], positions[0][3], positions[0][4])
            
            player_positions = {}
            for pos in positions[1:]:
                pid = pos[1]
                player_positions[pid] = (pos[2], pos[3])
            
            moments.append(Moment(
                quarter=quarter,
                timestamp=timestamp,
                game_clock=game_clock,
                shot_clock=shot_clock,
                ball_pos=ball_pos,
                player_positions=player_positions
            ))
    
    # Sort moments by timestamp
    moments.sort(key=lambda m: m.timestamp)
    
    # Load play-by-play
    events = load_play_by_play(game_id)
    
    # Link events to frame indices by finding closest matching moment
    for evt in events:
        if evt.game_clock_str and evt.period:
            try:
                target_clock = parse_game_clock(evt.game_clock_str)
                target_quarter = evt.period
                
                # Find the frame that best matches this event
                best_frame = 0
                best_diff = float('inf')
                
                for i, moment in enumerate(moments):
                    if moment.quarter == target_quarter:
                        diff = abs(moment.game_clock - target_clock)
                        if diff < best_diff:
                            best_diff = diff
                            best_frame = i
                
                evt.frame_index = best_frame
            except (ValueError, IndexError):
                pass
    
    # Filter events to only those with valid frame indices and sort by frame
    events = [e for e in events if e.frame_index >= 0]
    events.sort(key=lambda e: e.frame_index)
    
    print(f"Loaded {len(moments)} tracking frames and {len(events)} play-by-play events")
    
    return GameData(
        game_id=game_id,
        game_date=game_date,
        home_team_name=home_info['name'],
        home_team_abbr=home_info['abbreviation'],
        home_team_id=home_info['teamid'],
        away_team_name=away_info['name'],
        away_team_abbr=away_info['abbreviation'],
        away_team_id=away_info['teamid'],
        players=players,
        moments=moments,
        events=events
    )


# ============================================================================
# Court Drawing Functions
# ============================================================================

def court_to_screen(x: float, y: float) -> Tuple[int, int]:
    """Convert court coordinates (feet) to screen coordinates (pixels)."""
    screen_x = int(x * COURT_SCALE)
    screen_y = int(y * COURT_SCALE)
    return (screen_x, screen_y)


def draw_court(surface: pygame.Surface):
    """Draw an NBA basketball court."""
    
    # Draw hardwood floor pattern
    for i in range(0, COURT_WIDTH_PX, 20):
        color = HARDWOOD_DARK if (i // 20) % 2 == 0 else HARDWOOD_LIGHT
        pygame.draw.rect(surface, color, (i, 0, 20, COURT_HEIGHT_PX))
    
    # Court boundaries
    pygame.draw.rect(surface, COURT_LINE_COLOR, (0, 0, COURT_WIDTH_PX, COURT_HEIGHT_PX), 3)
    
    # Half court line
    half_x = COURT_WIDTH_PX // 2
    pygame.draw.line(surface, COURT_LINE_COLOR, (half_x, 0), (half_x, COURT_HEIGHT_PX), 3)
    
    # Center circle (radius 6 feet = 72 pixels)
    center_y = COURT_HEIGHT_PX // 2
    pygame.draw.circle(surface, CENTER_CIRCLE_COLOR, (half_x, center_y), int(6 * COURT_SCALE), 3)
    pygame.draw.circle(surface, CENTER_CIRCLE_COLOR, (half_x, center_y), int(2 * COURT_SCALE), 3)
    
    # Draw elements for both sides
    for side in [0, 1]:  # 0 = left, 1 = right
        if side == 0:
            base_x = 0
            direction = 1
        else:
            base_x = COURT_WIDTH_PX
            direction = -1
        
        # Basket position (4 feet from baseline)
        basket_x = base_x + direction * int(4 * COURT_SCALE)
        basket_y = center_y
        
        # Backboard (6 feet wide, 3.5 feet from baseline)
        backboard_x = base_x + direction * int(4 * COURT_SCALE)
        pygame.draw.line(surface, COURT_LINE_COLOR, 
                        (backboard_x, center_y - int(3 * COURT_SCALE)),
                        (backboard_x, center_y + int(3 * COURT_SCALE)), 4)
        
        # Rim (1.5 feet diameter)
        rim_x = base_x + direction * int(5.25 * COURT_SCALE)
        pygame.draw.circle(surface, THREE_POINT_COLOR, (rim_x, center_y), int(0.75 * COURT_SCALE), 3)
        
        # Paint/Key (16 feet wide, 19 feet long)
        key_width = int(16 * COURT_SCALE)
        key_length = int(19 * COURT_SCALE)
        key_top = center_y - key_width // 2
        
        if side == 0:
            key_rect = pygame.Rect(0, key_top, key_length, key_width)
        else:
            key_rect = pygame.Rect(COURT_WIDTH_PX - key_length, key_top, key_length, key_width)
        
        # Fill key with slightly different color
        pygame.draw.rect(surface, KEY_COLOR, key_rect)
        pygame.draw.rect(surface, COURT_LINE_COLOR, key_rect, 3)
        
        # Free throw circle (6 feet radius)
        ft_x = base_x + direction * key_length
        pygame.draw.circle(surface, COURT_LINE_COLOR, (ft_x, center_y), int(6 * COURT_SCALE), 2)
        
        # Restricted area arc (4 feet radius from basket)
        restricted_center_x = base_x + direction * int(5.25 * COURT_SCALE)
        if side == 0:
            pygame.draw.arc(surface, COURT_LINE_COLOR,
                          (restricted_center_x - int(4 * COURT_SCALE), 
                           center_y - int(4 * COURT_SCALE),
                           int(8 * COURT_SCALE), int(8 * COURT_SCALE)),
                          -1.57, 1.57, 2)
        else:
            pygame.draw.arc(surface, COURT_LINE_COLOR,
                          (restricted_center_x - int(4 * COURT_SCALE),
                           center_y - int(4 * COURT_SCALE),
                           int(8 * COURT_SCALE), int(8 * COURT_SCALE)),
                          1.57, 4.71, 2)
        
        # Three-point line
        three_radius = int(23.75 * COURT_SCALE)
        three_center_x = base_x + direction * int(5.25 * COURT_SCALE)
        
        corner_dist = int(3 * COURT_SCALE)  # 3 feet from sideline
        
        if side == 0:
            pygame.draw.line(surface, THREE_POINT_COLOR, 
                           (0, corner_dist), (int(14 * COURT_SCALE), corner_dist), 3)
            pygame.draw.line(surface, THREE_POINT_COLOR,
                           (0, COURT_HEIGHT_PX - corner_dist), 
                           (int(14 * COURT_SCALE), COURT_HEIGHT_PX - corner_dist), 3)
            arc_rect = pygame.Rect(three_center_x - three_radius, 
                                   center_y - three_radius,
                                   three_radius * 2, three_radius * 2)
            pygame.draw.arc(surface, THREE_POINT_COLOR, arc_rect, -1.2, 1.2, 3)
        else:
            pygame.draw.line(surface, THREE_POINT_COLOR,
                           (COURT_WIDTH_PX, corner_dist), 
                           (COURT_WIDTH_PX - int(14 * COURT_SCALE), corner_dist), 3)
            pygame.draw.line(surface, THREE_POINT_COLOR,
                           (COURT_WIDTH_PX, COURT_HEIGHT_PX - corner_dist),
                           (COURT_WIDTH_PX - int(14 * COURT_SCALE), COURT_HEIGHT_PX - corner_dist), 3)
            arc_rect = pygame.Rect(three_center_x - three_radius,
                                   center_y - three_radius,
                                   three_radius * 2, three_radius * 2)
            pygame.draw.arc(surface, THREE_POINT_COLOR, arc_rect, 1.94, 4.34, 3)


# ============================================================================
# Game Renderer
# ============================================================================

class GameRenderer:
    def __init__(self, game_data: GameData):
        pygame.init()
        pygame.font.init()
        
        self.game = game_data
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(
            f"NBA Game Replay: {game_data.away_team_abbr} @ {game_data.home_team_abbr} - {game_data.game_date}"
        )
        
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)
        self.font_micro = pygame.font.Font(None, 16)
        
        # Pre-render court surface (only the court area, not the analysis panel)
        self.court_surface = pygame.Surface((COURT_WIDTH_PX, COURT_HEIGHT_PX))
        draw_court(self.court_surface)
        
        # Game state
        self.current_frame = 0
        self.running = True
        
        # Playback state: 'stopped', 'forward', 'reverse'
        self.playback_state = 'stopped'
        
        # Event navigation
        self.current_event_index = -1  # Start before first event
        self.next_event_frame = 0  # Frame of next event (for forward play)
        self.prev_event_frame = 0  # Frame of previous event (for reverse play)
        
        # Score tracking
        self.home_score = 0
        self.away_score = 0
        
        # Event log (rolling text) - current + 4 past events
        self.event_log: deque = deque(maxlen=5)
        
        # Shot celebration display
        self.shot_display_text = ""
        self.shot_display_timer = 0
        
        # Position analysis
        self.current_analysis: Optional[PositionAnalysis] = None
        
        # UI Components - positioned in the court area (left side)
        control_y = COURT_HEIGHT_PX + 60
        court_center_x = COURT_WIDTH_PX // 2
        
        # Play/Pause button
        self.play_btn = Button(
            court_center_x - 70, control_y, 140, 40, "▶ Play"
        )
        
        # Reverse Play button
        self.reverse_btn = Button(
            court_center_x - 220, control_y, 140, 40, "◀ Reverse"
        )
        
        # Stop button
        self.stop_btn = Button(
            court_center_x + 80, control_y, 100, 40, "■ Stop"
        )
        
        # Speed slider
        self.speed_slider = SpeedSlider(
            50, control_y + 60, 350, 50
        )
        
        # Track processed events for score
        self.processed_events = set()
        
        # Pass chain tracking
        self.current_ball_carrier_id: Optional[int] = None
        self.current_ball_carrier_pos: Tuple[float, float] = (0, 0)
        self.pass_chain = PassChain()
        self.last_completed_chain: Optional[PassChain] = None
        
        # Minimum distance threshold to consider ball possession changed (in feet)
        self.possession_threshold = 3.0
        
        # Ball trail - stores (position, team_id) tuples for colored line trail
        # Each entry is ((x, y, z), team_id) where team_id is the team with possession at that moment
        self.ball_trail: deque = deque(maxlen=150)  # Store many positions for long trail
        
        # Shot result tracking (simplified - no trajectory)
        self.showing_shot_result = False
        self.pending_shot_event: Optional[PlayByPlayEvent] = None
    
    def get_event_description(self, event: PlayByPlayEvent) -> str:
        """Generate a description string for an event."""
        desc = event.home_desc or event.visitor_desc or event.neutral_desc
        if desc:
            return desc
        
        # Fallback to event type description
        event_name = event_dict.get(event.event_type, 'unknown')
        if event_name in event_desc:
            action_desc = event_desc[event_name].get(event.action_type, '')
            if action_desc:
                player = event.player1_name or ''
                return f"{player} - {action_desc}"
        
        return ""
    
    def process_event(self, event: PlayByPlayEvent):
        """Process an event - update score, add to log, show celebration."""
        if event.event_num in self.processed_events:
            return
        
        self.processed_events.add(event.event_num)
        
        # Update score
        if event.score:
            parts = event.score.split(' - ')
            if len(parts) == 2:
                try:
                    self.away_score = int(parts[0])
                    self.home_score = int(parts[1])
                except ValueError:
                    pass
        
        # Get event description
        desc = self.get_event_description(event)
        if desc:
            # Add quarter info
            q_str = f"Q{event.period}" if event.period <= 4 else f"OT{event.period - 4}"
            full_desc = f"[{q_str} {event.game_clock_str}] {desc}"
            self.event_log.append(full_desc)
        
        # Handle shot events (made or missed) - publish pass chain
        if event.event_type in [1, 2]:  # Shot made (1) or missed (2)
            player_name = event.player1_name or "Unknown"
            team_abbr = event.player1_team_abbr or ""
            
            # Complete the pass chain with shooter info
            self.pass_chain.shooter_id = event.player1_id
            self.pass_chain.shooter_name = player_name
            self.pass_chain.shot_made = (event.event_type == 1)
            
            if self.current_ball_carrier_pos:
                self.pass_chain.shooter_pos = self.current_ball_carrier_pos
            
            # Store completed chain and start fresh
            if self.pass_chain.total_passes > 0 or self.pass_chain.shooter_id:
                self.last_completed_chain = PassChain(
                    passes=list(self.pass_chain.passes),
                    shooter_id=self.pass_chain.shooter_id,
                    shooter_name=self.pass_chain.shooter_name,
                    shooter_pos=self.pass_chain.shooter_pos,
                    shot_made=self.pass_chain.shot_made
                )
                
                # Print pass chain to console
                self.print_pass_chain(self.last_completed_chain)
            
            # Clear for next possession
            self.pass_chain.clear()
            
            if event.event_type == 1:  # Shot made
                self.shot_display_text = f"🏀 {player_name} ({team_abbr}) SCORES!"
                self.shot_display_timer = 90
        
        # Reset pass chain on turnovers, rebounds, etc.
        elif event.event_type in [4, 5]:  # Rebound or Turnover
            self.pass_chain.clear()
    
    def print_pass_chain(self, chain: PassChain):
        """Print the pass chain to console."""
        result = "MADE" if chain.shot_made else "MISSED"
        print(f"\n{'='*60}")
        print(f"SHOT {result} by {chain.shooter_name}")
        print(f"  Shooter position: ({chain.shooter_pos[0]:.1f}, {chain.shooter_pos[1]:.1f})")
        print(f"  Total passes: {chain.total_passes}")
        print(f"  Total pass distance: {chain.total_distance:.1f} ft")
        
        if chain.passes:
            print("  Pass sequence:")
            for i, p in enumerate(chain.passes, 1):
                print(f"    {i}. #{p.passer_jersey} {p.passer_name} -> #{p.receiver_jersey} {p.receiver_name}")
                print(f"       From ({p.passer_pos[0]:.1f}, {p.passer_pos[1]:.1f}) to ({p.receiver_pos[0]:.1f}, {p.receiver_pos[1]:.1f})")
                print(f"       Distance: {p.pass_distance:.1f} ft")
        print(f"{'='*60}\n")
    
    def update_ball_carrier(self, moment: Moment):
        """Track ball carrier changes and detect passes."""
        ball_pos = moment.ball_pos
        ball_xy = (ball_pos[0], ball_pos[1])
        
        # Find current ball carrier (closest player to ball)
        min_dist = float('inf')
        new_carrier_id = None
        new_carrier_pos = (0, 0)
        
        for pid, pos in moment.player_positions.items():
            dist = calculate_distance(ball_xy, pos)
            if dist < min_dist:
                min_dist = dist
                new_carrier_id = pid
                new_carrier_pos = pos
        
        # Only consider possession if player is close enough to ball
        if min_dist > self.possession_threshold:
            return  # Ball is in the air or not possessed
        
        # Check if carrier changed (potential pass)
        if (new_carrier_id and 
            self.current_ball_carrier_id and 
            new_carrier_id != self.current_ball_carrier_id):
            
            # Verify both players are on the same team (it's a pass, not a steal)
            if (new_carrier_id in self.game.players and 
                self.current_ball_carrier_id in self.game.players):
                
                new_player = self.game.players[new_carrier_id]
                old_player = self.game.players[self.current_ball_carrier_id]
                
                if new_player.team_id == old_player.team_id:
                    # This is a pass!
                    pass_event = PassEvent(
                        passer_id=self.current_ball_carrier_id,
                        passer_name=old_player.display_name,
                        passer_jersey=old_player.jersey,
                        passer_pos=self.current_ball_carrier_pos,
                        receiver_id=new_carrier_id,
                        receiver_name=new_player.display_name,
                        receiver_jersey=new_player.jersey,
                        receiver_pos=new_carrier_pos,
                        timestamp=moment.timestamp
                    )
                    self.pass_chain.add_pass(pass_event)
                else:
                    # Possession changed teams - clear pass chain
                    self.pass_chain.clear()
        
        # Update current carrier
        self.current_ball_carrier_id = new_carrier_id
        self.current_ball_carrier_pos = new_carrier_pos
    
    def find_next_event_frame(self) -> Tuple[int, int]:
        """Find the next event after current frame. Returns (event_index, frame_index)."""
        for i, event in enumerate(self.game.events):
            if event.frame_index > self.current_frame:
                return i, event.frame_index
        return len(self.game.events) - 1, len(self.game.moments) - 1
    
    def find_prev_event_frame(self) -> Tuple[int, int]:
        """Find the previous event before current frame. Returns (event_index, frame_index)."""
        for i in range(len(self.game.events) - 1, -1, -1):
            if self.game.events[i].frame_index < self.current_frame:
                return i, self.game.events[i].frame_index
        return 0, 0
    
    def play_forward(self):
        """Start playing forward until next event."""
        # If showing shot result, acknowledge and continue
        if self.showing_shot_result:
            self.showing_shot_result = False
            pending_event = self.pending_shot_event
            self.pending_shot_event = None
            
            # Process the event now
            if pending_event:
                self.process_event(pending_event)
                # Move frame to event frame so we don't re-trigger the same event
                self.current_frame = pending_event.frame_index
            
            # Now continue to find the NEXT event after this one
            # Don't return - fall through to start playing to next event
        
        if self.playback_state == 'forward':
            # Already playing forward - pause
            self.stop_playback()
        else:
            # Find next event
            next_idx, next_frame = self.find_next_event_frame()
            if next_frame > self.current_frame:
                self.current_event_index = next_idx
                event = self.game.events[next_idx]
                
                # Check if this is a shot event - show result when we reach it
                if event.event_type in [1, 2]:  # Shot made or missed
                    self.pending_shot_event = event
                else:
                    self.pending_shot_event = None
                
                self.next_event_frame = next_frame
                self.playback_state = 'forward'
                print(f"Playing forward to event {next_idx + 1}")
    
    def play_reverse(self):
        """Start playing in reverse until previous event."""
        # If showing shot result, dismiss it first
        if self.showing_shot_result:
            self.showing_shot_result = False
            self.pending_shot_event = None
        
        if self.playback_state == 'reverse':
            # Already playing reverse - pause
            self.stop_playback()
        else:
            self.pending_shot_event = None
            
            # Find previous event
            prev_idx, prev_frame = self.find_prev_event_frame()
            if prev_frame < self.current_frame:
                self.prev_event_frame = prev_frame
                self.current_event_index = prev_idx
                self.playback_state = 'reverse'
                print(f"Playing reverse to event {prev_idx + 1}")
    
    def stop_playback(self):
        """Stop all playback."""
        self.playback_state = 'stopped'
        
        # Clear pending shot unless we're showing shot result
        if not self.showing_shot_result:
            self.pending_shot_event = None
        
        # Calculate analysis at current position
        if self.current_frame < len(self.game.moments):
            moment = self.game.moments[self.current_frame]
            self.current_analysis = analyze_positions(
                moment,
                self.game.players,
                self.game.home_team_id,
                self.game.away_team_id
            )
    
    def on_event_reached(self, event_index: int, direction: str):
        """Called when playback reaches an event."""
        if 0 <= event_index < len(self.game.events):
            event = self.game.events[event_index]
            
            # Check if this is a shot event - show result overlay
            if direction == 'forward' and self.pending_shot_event:
                self.showing_shot_result = True
                # Don't process event yet - wait for user to acknowledge
            else:
                if direction == 'forward':
                    self.process_event(event)
            
            # Calculate analysis
            moment = self.game.moments[self.current_frame]
            self.current_analysis = analyze_positions(
                moment,
                self.game.players,
                self.game.home_team_id,
                self.game.away_team_id
            )
        self.playback_state = 'stopped'
    
    def draw_player(self, player_id: int, x: float, y: float):
        """Draw a player circle with jersey number."""
        if player_id not in self.game.players:
            return
        
        player = self.game.players[player_id]
        screen_pos = court_to_screen(x, y)
        
        # Determine team color
        if player.team_id == self.game.home_team_id:
            color = HOME_COLOR
        else:
            color = AWAY_COLOR
        
        # Draw player circle with border
        pygame.draw.circle(self.screen, (0, 0, 0), screen_pos, PLAYER_RADIUS + 2)
        pygame.draw.circle(self.screen, color, screen_pos, PLAYER_RADIUS)
        
        # Draw jersey number
        jersey_text = self.font_tiny.render(player.jersey, True, (255, 255, 255))
        text_rect = jersey_text.get_rect(center=screen_pos)
        self.screen.blit(jersey_text, text_rect)
    
    def draw_ball(self, x: float, y: float, z: float):
        """Draw the basketball with height indication and team color based on possession."""
        screen_pos = court_to_screen(x, y)
        
        # Ball size varies slightly with height for depth perception
        size = int(BALL_RADIUS + min(z * 0.5, 6))
        
        # Determine ball color based on possession
        ball_color = BALL_COLOR  # Default orange
        ball_outline = BALL_SHADOW
        if self.current_ball_carrier_id and self.current_ball_carrier_id in self.game.players:
            carrier = self.game.players[self.current_ball_carrier_id]
            if carrier.team_id == self.game.home_team_id:
                ball_color = HOME_COLOR
                ball_outline = (max(0, HOME_COLOR[0] - 50), max(0, HOME_COLOR[1] - 50), max(0, HOME_COLOR[2] - 50))
            else:
                ball_color = AWAY_COLOR
                ball_outline = (max(0, AWAY_COLOR[0] - 50), max(0, AWAY_COLOR[1] - 50), max(0, AWAY_COLOR[2] - 50))
        
        # Shadow
        shadow_offset = int(z * 0.3)
        pygame.draw.circle(self.screen, (50, 50, 50), 
                          (screen_pos[0] + 3, screen_pos[1] + 3 + shadow_offset), 
                          size - 2)
        
        # Ball with team color
        pygame.draw.circle(self.screen, ball_color, screen_pos, size)
        pygame.draw.circle(self.screen, ball_outline, screen_pos, size, 2)
        
        # White highlight for visibility
        pygame.draw.circle(self.screen, (255, 255, 255), 
                          (screen_pos[0] - size//3, screen_pos[1] - size//3), 
                          size//3)
    
    def draw_scoreboard(self, moment: Moment):
        """Draw the scoreboard header."""
        # Header background
        header_rect = pygame.Rect(0, COURT_HEIGHT_PX, WINDOW_WIDTH, 55)
        pygame.draw.rect(self.screen, HEADER_COLOR, header_rect)
        
        # Game clock
        q_str = f"Q{moment.quarter}" if moment.quarter <= 4 else f"OT{moment.quarter - 4}"
        clock_min = int(moment.game_clock // 60)
        clock_sec = int(moment.game_clock % 60)
        clock_str = f"{q_str}  {clock_min}:{clock_sec:02d}"
        
        clock_text = self.font_large.render(clock_str, True, TEXT_COLOR)
        self.screen.blit(clock_text, (WINDOW_WIDTH // 2 - clock_text.get_width() // 2, 
                                       COURT_HEIGHT_PX + 8))
        
        # Shot clock
        if moment.shot_clock and moment.shot_clock > 0:
            sc_text = self.font_medium.render(f"Shot: {moment.shot_clock:.1f}", True, (200, 200, 200))
            self.screen.blit(sc_text, (WINDOW_WIDTH - 140, COURT_HEIGHT_PX + 12))
        
        # Away team score (left)
        away_text = self.font_medium.render(f"{self.game.away_team_abbr}", True, AWAY_COLOR)
        away_score_text = self.font_large.render(f"{self.away_score}", True, SCORE_COLOR)
        self.screen.blit(away_text, (30, COURT_HEIGHT_PX + 5))
        self.screen.blit(away_score_text, (30, COURT_HEIGHT_PX + 28))
        
        # Home team score
        home_text = self.font_medium.render(f"{self.game.home_team_abbr}", True, HOME_COLOR)
        home_score_text = self.font_large.render(f"{self.home_score}", True, SCORE_COLOR)
        self.screen.blit(home_text, (130, COURT_HEIGHT_PX + 5))
        self.screen.blit(home_score_text, (130, COURT_HEIGHT_PX + 28))
        
        # Event counter
        event_str = f"Event: {self.current_event_index + 1}/{len(self.game.events)}"
        event_text = self.font_small.render(event_str, True, (180, 180, 180))
        self.screen.blit(event_text, (WINDOW_WIDTH - 130, COURT_HEIGHT_PX + 38))
    
    def draw_controls(self):
        """Draw control panel with buttons and slider."""
        # Control area background
        control_top = COURT_HEIGHT_PX + 55
        pygame.draw.rect(self.screen, EVENT_BG_COLOR, 
                        (0, control_top, COURT_WIDTH_PX, WINDOW_HEIGHT - control_top))
        
        # Update button text based on state
        if self.showing_shot_result:
            self.play_btn.text = "▶ Continue"
        elif self.playback_state == 'forward':
            self.play_btn.text = "⏸ Pause"
        else:
            self.play_btn.text = "▶ Play"
        
        if self.playback_state == 'reverse':
            self.reverse_btn.text = "⏸ Pause"
        else:
            self.reverse_btn.text = "◀ Reverse"
        
        # Draw buttons
        self.reverse_btn.draw(self.screen)
        self.play_btn.draw(self.screen)
        self.stop_btn.draw(self.screen)
        
        # Draw slider
        self.speed_slider.draw(self.screen)
        
        # Playback status - positioned below the buttons, left of the event log
        if self.playback_state == 'forward':
            status = "▶ Playing forward..."
            status_color = (100, 255, 100)
        elif self.playback_state == 'reverse':
            status = "◀ Playing reverse..."
            status_color = (100, 200, 255)
        else:
            status = "⏹ Stopped - Press Play or Reverse"
            status_color = (255, 200, 100)
        
        status_text = self.font_small.render(status, True, status_color)
        self.screen.blit(status_text, (50, COURT_HEIGHT_PX + 185))
    
    def draw_event_log(self):
        """Draw the rolling event log with most recent at top."""
        log_top = COURT_HEIGHT_PX + 170
        
        # Title
        title = self.font_small.render("PLAY-BY-PLAY", True, (180, 180, 180))
        self.screen.blit(title, (450, log_top))
        
        # Events - reversed so newest is at top
        y_offset = log_top + 22
        events_reversed = list(reversed(self.event_log))
        for i, event_text in enumerate(events_reversed):
            # Truncate if too long
            if len(event_text) > 70:
                event_text = event_text[:67] + "..."
            
            # Fade older events
            if i == 0:
                color = SCORE_COLOR  # Current event highlighted
            else:
                fade = max(120, 240 - i * 30)
                color = (fade, fade, fade)
            
            text_surface = self.font_small.render(event_text, True, color)
            self.screen.blit(text_surface, (450, y_offset))
            y_offset += 20
    
    def draw_ball_trail(self):
        """Draw a line trail behind the ball showing its path, colored by possession."""
        if len(self.ball_trail) < 2:
            return
        
        # Draw trail in segments, changing color when possession changes
        trail_list = list(self.ball_trail)
        
        current_segment = []
        current_team = trail_list[0][1]  # team_id from first entry
        
        for pos, team_id in trail_list:
            screen_pos = court_to_screen(pos[0], pos[1])
            
            if team_id != current_team:
                # Team changed - draw current segment and start new one
                if len(current_segment) >= 2:
                    # Get color for current team
                    if current_team == self.game.home_team_id:
                        color = HOME_COLOR
                    elif current_team == self.game.away_team_id:
                        color = AWAY_COLOR
                    else:
                        color = BALL_COLOR  # No possession - orange
                    pygame.draw.lines(self.screen, color, False, current_segment, 2)
                
                # Start new segment (include last point of previous for continuity)
                current_segment = [current_segment[-1]] if current_segment else []
                current_segment.append(screen_pos)
                current_team = team_id
            else:
                current_segment.append(screen_pos)
        
        # Draw final segment
        if len(current_segment) >= 2:
            if current_team == self.game.home_team_id:
                color = HOME_COLOR
            elif current_team == self.game.away_team_id:
                color = AWAY_COLOR
            else:
                color = BALL_COLOR
            pygame.draw.lines(self.screen, color, False, current_segment, 2)
    
    def draw_pending_shot_info(self):
        """Draw info about the pending shot event."""
        if not self.showing_shot_result or not self.pending_shot_event:
            return
        
        event = self.pending_shot_event
        
        # Determine result
        if event.event_type == 1:
            result_text = "SHOT MADE!"
            result_color = (100, 255, 100)
        else:
            result_text = "SHOT MISSED"
            result_color = (255, 100, 100)
        
        # Get description
        desc = self.get_event_description(event)
        
        # Draw overlay box
        box_width = 500
        box_height = 120
        box_x = (COURT_WIDTH_PX - box_width) // 2
        box_y = COURT_HEIGHT_PX // 2 - box_height // 2
        
        pygame.draw.rect(self.screen, (20, 25, 40), 
                        (box_x, box_y, box_width, box_height), border_radius=15)
        pygame.draw.rect(self.screen, result_color, 
                        (box_x, box_y, box_width, box_height), 4, border_radius=15)
        
        # Result text
        result_surface = self.font_large.render(result_text, True, result_color)
        result_rect = result_surface.get_rect(center=(box_x + box_width // 2, box_y + 30))
        self.screen.blit(result_surface, result_rect)
        
        # Description
        if desc:
            desc_surface = self.font_small.render(desc[:60], True, TEXT_COLOR)
            desc_rect = desc_surface.get_rect(center=(box_x + box_width // 2, box_y + 65))
            self.screen.blit(desc_surface, desc_rect)
        
        # Instruction
        instr_text = "Press SPACE or Play to continue"
        instr_surface = self.font_tiny.render(instr_text, True, (150, 150, 150))
        instr_rect = instr_surface.get_rect(center=(box_x + box_width // 2, box_y + 95))
        self.screen.blit(instr_surface, instr_rect)
    
    def draw_shot_celebration(self):
        """Draw shot made celebration overlay."""
        if self.shot_display_timer > 0:
            # Draw text with glow effect
            text = self.font_large.render(self.shot_display_text, True, SCORE_COLOR)
            text_rect = text.get_rect(center=(COURT_WIDTH_PX // 2, COURT_HEIGHT_PX // 2))
            
            # Background box
            bg_rect = text_rect.inflate(40, 20)
            pygame.draw.rect(self.screen, (30, 30, 50), bg_rect, border_radius=10)
            pygame.draw.rect(self.screen, SCORE_COLOR, bg_rect, 3, border_radius=10)
            
            self.screen.blit(text, text_rect)
            self.shot_display_timer -= 1
    
    def draw_legend(self):
        """Draw team color legend."""
        legend_y = COURT_HEIGHT_PX - 25
        
        # Away team
        pygame.draw.circle(self.screen, AWAY_COLOR, (15, legend_y), 8)
        away_label = self.font_tiny.render(self.game.away_team_name, True, TEXT_COLOR)
        self.screen.blit(away_label, (28, legend_y - 7))
        
        # Home team
        home_x = 200
        pygame.draw.circle(self.screen, HOME_COLOR, (home_x, legend_y), 8)
        home_label = self.font_tiny.render(self.game.home_team_name, True, TEXT_COLOR)
        self.screen.blit(home_label, (home_x + 13, legend_y - 7))
    
    def get_team_colors(self, analysis: PositionAnalysis) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get offensive and defensive team colors based on who has the ball."""
        if analysis.offensive_team_id == self.game.home_team_id:
            return HOME_COLOR, AWAY_COLOR
        else:
            return AWAY_COLOR, HOME_COLOR
    
    def draw_analysis_panel(self):
        """Draw the vector analysis panel on the right side."""
        panel_x = COURT_WIDTH_PX
        panel_width = ANALYSIS_PANEL_WIDTH
        panel_height = WINDOW_HEIGHT
        
        # Panel background
        pygame.draw.rect(self.screen, ANALYSIS_BG_COLOR, 
                        (panel_x, 0, panel_width, panel_height))
        pygame.draw.line(self.screen, (60, 60, 80), 
                        (panel_x, 0), (panel_x, panel_height), 2)
        
        # Title
        title = self.font_medium.render("POSITION ANALYSIS", True, SCORE_COLOR)
        self.screen.blit(title, (panel_x + 15, 10))
        
        if not self.current_analysis or not self.current_analysis.ball_carrier_id:
            no_data = self.font_small.render("Waiting for event...", True, (120, 120, 120))
            self.screen.blit(no_data, (panel_x + 15, 50))
            self._draw_pass_chain_section(panel_x, 100)
            return
        
        analysis = self.current_analysis
        off_color, def_color = self.get_team_colors(analysis)
        y = 45
        
        # Ball carrier info
        carrier_text = f"Ball Carrier: {analysis.ball_carrier_name}"
        self.screen.blit(self.font_small.render(carrier_text, True, TEXT_COLOR), (panel_x + 15, y))
        y += 25
        
        # Determine offensive team name and show with colors
        if analysis.offensive_team_id == self.game.home_team_id:
            off_team = self.game.home_team_abbr
            def_team = self.game.away_team_abbr
        else:
            off_team = self.game.away_team_abbr
            def_team = self.game.home_team_abbr
        
        # Draw team labels with their colors
        off_label = self.font_tiny.render(f"Offense: {off_team}", True, off_color)
        def_label = self.font_tiny.render(f"Defense: {def_team}", True, def_color)
        self.screen.blit(off_label, (panel_x + 15, y))
        self.screen.blit(def_label, (panel_x + 130, y))
        y += 28
        
        # ===== VECTOR DIAGRAM =====
        diagram_x = panel_x + 20
        diagram_y = y
        diagram_size = 160
        diagram_center = (diagram_x + diagram_size // 2, diagram_y + diagram_size // 2)
        
        # Diagram background
        pygame.draw.rect(self.screen, (30, 35, 45), 
                        (diagram_x, diagram_y, diagram_size, diagram_size), border_radius=5)
        pygame.draw.rect(self.screen, (60, 60, 80), 
                        (diagram_x, diagram_y, diagram_size, diagram_size), 2, border_radius=5)
        
        # Draw axes
        pygame.draw.line(self.screen, (60, 60, 80), 
                        (diagram_center[0], diagram_y + 10), 
                        (diagram_center[0], diagram_y + diagram_size - 10), 1)
        pygame.draw.line(self.screen, (60, 60, 80), 
                        (diagram_x + 10, diagram_center[1]), 
                        (diagram_x + diagram_size - 10, diagram_center[1]), 1)
        
        # Scale factor for diagram (max distance shown = 50 feet)
        scale = (diagram_size / 2 - 15) / 50.0
        
        # Draw carrier at center (ball carrier is origin) - with offensive team color
        pygame.draw.circle(self.screen, off_color, diagram_center, 8)
        pygame.draw.circle(self.screen, BALL_COLOR, diagram_center, 5)  # Ball in center
        
        # Draw hoop direction vector - colored with offensive team color
        if analysis.carrier_to_hoop:
            hoop_vec = analysis.carrier_to_hoop.normalized()
            hoop_end = (
                int(diagram_center[0] + hoop_vec.dx * 55),
                int(diagram_center[1] + hoop_vec.dy * 55)
            )
            pygame.draw.line(self.screen, off_color, diagram_center, hoop_end, 3)
            # Hoop indicator - colored with offensive team color
            pygame.draw.circle(self.screen, off_color, hoop_end, 8)
            pygame.draw.circle(self.screen, (255, 255, 255), hoop_end, 8, 2)
            pygame.draw.circle(self.screen, (255, 255, 255), hoop_end, 3)  # Inner dot
        
        # Draw offensive players (circles) - relative to carrier
        for pv in analysis.offense_to_carrier[:4]:
            if pv.distance > 0:
                px = int(diagram_center[0] + pv.vector.dx * scale)
                py = int(diagram_center[1] + pv.vector.dy * scale)
                px = max(diagram_x + 8, min(diagram_x + diagram_size - 8, px))
                py = max(diagram_y + 8, min(diagram_y + diagram_size - 8, py))
                pygame.draw.circle(self.screen, off_color, (px, py), 5)
                pygame.draw.line(self.screen, off_color, diagram_center, (px, py), 1)
        
        # Draw defensive players (squares) - relative to carrier
        for pv in analysis.defense_to_carrier[:5]:
            if pv.distance > 0:
                px = int(diagram_center[0] + pv.vector.dx * scale)
                py = int(diagram_center[1] + pv.vector.dy * scale)
                px = max(diagram_x + 8, min(diagram_x + diagram_size - 8, px))
                py = max(diagram_y + 8, min(diagram_y + diagram_size - 8, py))
                # Draw square for defense
                pygame.draw.rect(self.screen, def_color, (px - 4, py - 4, 8, 8))
                pygame.draw.line(self.screen, def_color, diagram_center, (px, py), 1)
        
        # Draw centroids with team colors and different shapes
        # Offensive centroid - circle with crosshair
        if analysis.offensive_centroid != (0, 0):
            off_centroid_rel = (
                analysis.offensive_centroid[0] - analysis.ball_carrier_pos[0],
                analysis.offensive_centroid[1] - analysis.ball_carrier_pos[1]
            )
            cx = int(diagram_center[0] + off_centroid_rel[0] * scale)
            cy = int(diagram_center[1] + off_centroid_rel[1] * scale)
            cx = max(diagram_x + 10, min(diagram_x + diagram_size - 10, cx))
            cy = max(diagram_y + 10, min(diagram_y + diagram_size - 10, cy))
            pygame.draw.circle(self.screen, off_color, (cx, cy), 7, 2)
            pygame.draw.line(self.screen, off_color, (cx - 5, cy), (cx + 5, cy), 2)
            pygame.draw.line(self.screen, off_color, (cx, cy - 5), (cx, cy + 5), 2)
        
        # Defensive centroid - square with X
        if analysis.defensive_centroid != (0, 0):
            def_centroid_rel = (
                analysis.defensive_centroid[0] - analysis.ball_carrier_pos[0],
                analysis.defensive_centroid[1] - analysis.ball_carrier_pos[1]
            )
            dx = int(diagram_center[0] + def_centroid_rel[0] * scale)
            dy = int(diagram_center[1] + def_centroid_rel[1] * scale)
            dx = max(diagram_x + 10, min(diagram_x + diagram_size - 10, dx))
            dy = max(diagram_y + 10, min(diagram_y + diagram_size - 10, dy))
            pygame.draw.rect(self.screen, def_color, (dx - 6, dy - 6, 12, 12), 2)
            pygame.draw.line(self.screen, def_color, (dx - 4, dy - 4), (dx + 4, dy + 4), 2)
            pygame.draw.line(self.screen, def_color, (dx + 4, dy - 4), (dx - 4, dy + 4), 2)
        
        y = diagram_y + diagram_size + 8
        
        # Legend for diagram - now with team colors
        # Row 1
        pygame.draw.circle(self.screen, off_color, (panel_x + 20, y + 7), 5)
        self.screen.blit(self.font_micro.render(f"Offense ({off_team})", True, off_color), (panel_x + 30, y))
        pygame.draw.rect(self.screen, def_color, (panel_x + 130, y + 3, 8, 8))
        self.screen.blit(self.font_micro.render(f"Defense ({def_team})", True, def_color), (panel_x + 142, y))
        y += 16
        
        # Row 2 - centroids
        pygame.draw.circle(self.screen, off_color, (panel_x + 20, y + 7), 5, 1)
        self.screen.blit(self.font_micro.render("Off. Centroid", True, (150, 150, 150)), (panel_x + 30, y))
        pygame.draw.rect(self.screen, def_color, (panel_x + 130, y + 3, 8, 8), 1)
        self.screen.blit(self.font_micro.render("Def. Centroid", True, (150, 150, 150)), (panel_x + 142, y))
        y += 20
        
        # ===== VECTOR DATA =====
        self.screen.blit(self.font_small.render("VECTOR DATA", True, SCORE_COLOR), (panel_x + 15, y))
        y += 20
        
        # Ball to hoop - use offensive team color
        if analysis.carrier_to_hoop:
            dist = analysis.carrier_to_hoop.magnitude
            angle = analysis.carrier_to_hoop.angle_degrees
            text = f"To Hoop: {dist:.1f}ft @ {angle:.0f}°"
            self.screen.blit(self.font_tiny.render(text, True, off_color), (panel_x + 15, y))
            y += 16
        
        # Centroids - use team colors
        if analysis.offensive_centroid != (0, 0):
            ox, oy = analysis.offensive_centroid
            text = f"Off. Centroid: ({ox:.1f}, {oy:.1f})"
            self.screen.blit(self.font_tiny.render(text, True, off_color), (panel_x + 15, y))
            y += 14
        
        if analysis.defensive_centroid != (0, 0):
            dx_val, dy_val = analysis.defensive_centroid
            text = f"Def. Centroid: ({dx_val:.1f}, {dy_val:.1f})"
            self.screen.blit(self.font_tiny.render(text, True, def_color), (panel_x + 15, y))
            y += 18
        
        # ===== NEAREST PLAYERS =====
        self.screen.blit(self.font_small.render("NEAREST PLAYERS", True, SCORE_COLOR), (panel_x + 15, y))
        y += 18
        
        # Offensive teammates
        self.screen.blit(self.font_tiny.render("Teammates:", True, off_color), (panel_x + 15, y))
        y += 14
        for pv in analysis.offense_to_carrier[:2]:
            text = f"  #{pv.jersey}: {pv.distance:.1f}ft"
            self.screen.blit(self.font_micro.render(text, True, (180, 180, 180)), (panel_x + 15, y))
            y += 12
        
        # Defenders
        self.screen.blit(self.font_tiny.render("Defenders:", True, def_color), (panel_x + 15, y))
        y += 14
        for pv in analysis.defense_to_carrier[:2]:
            text = f"  #{pv.jersey}: {pv.distance:.1f}ft"
            self.screen.blit(self.font_micro.render(text, True, (180, 180, 180)), (panel_x + 15, y))
            y += 12
        
        y += 5
        
        # ===== PASS CHAIN =====
        self._draw_pass_chain_section(panel_x, y)
    
    def _draw_pass_chain_section(self, panel_x: int, y: int):
        """Draw the pass chain tracking section."""
        self.screen.blit(self.font_small.render("PASS CHAIN", True, PASS_CHAIN_COLOR), (panel_x + 15, y))
        y += 20
        
        # Current possession chain
        if self.pass_chain.total_passes > 0:
            chain_text = f"Current: {self.pass_chain.total_passes} passes"
            self.screen.blit(self.font_tiny.render(chain_text, True, TEXT_COLOR), (panel_x + 15, y))
            y += 16
            
            # Show last few passes in current chain
            for p in self.pass_chain.passes[-3:]:
                text = f"  #{p.passer_jersey} → #{p.receiver_jersey}"
                self.screen.blit(self.font_micro.render(text, True, (180, 180, 180)), (panel_x + 15, y))
                y += 12
        else:
            self.screen.blit(self.font_tiny.render("No passes yet", True, (100, 100, 100)), (panel_x + 15, y))
            y += 16
        
        y += 8
        
        # Last completed chain (from last shot)
        if self.last_completed_chain:
            chain = self.last_completed_chain
            result_color = (100, 255, 100) if chain.shot_made else (255, 100, 100)
            result_text = "MADE" if chain.shot_made else "MISSED"
            
            header = f"Last Shot: {result_text}"
            self.screen.blit(self.font_tiny.render(header, True, result_color), (panel_x + 15, y))
            y += 14
            
            shooter_text = f"  Shooter: {chain.shooter_name}"
            self.screen.blit(self.font_micro.render(shooter_text, True, (180, 180, 180)), (panel_x + 15, y))
            y += 12
            
            stats_text = f"  {chain.total_passes} passes, {chain.total_distance:.1f}ft"
            self.screen.blit(self.font_micro.render(stats_text, True, (150, 150, 150)), (panel_x + 15, y))
            y += 14
            
            # Show pass sequence
            if chain.passes:
                self.screen.blit(self.font_micro.render("  Sequence:", True, (150, 150, 150)), (panel_x + 15, y))
                y += 12
                for i, p in enumerate(chain.passes[-4:], 1):
                    text = f"    {i}. #{p.passer_jersey}→#{p.receiver_jersey} ({p.pass_distance:.0f}ft)"
                    self.screen.blit(self.font_micro.render(text, True, (140, 140, 140)), (panel_x + 15, y))
                    y += 11
    
    def run(self):
        """Main game loop."""
        print("Starting game replay...")
        print("Controls:")
        print("  SPACE or → : Play forward to next event")
        print("  ← : Play reverse to previous event")
        print("  S : Stop playback")
        print("  Q : Quit")
        print("Use the speed slider to adjust playback speed")
        
        while self.running and self.current_frame < len(self.game.moments):
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_SPACE:
                        self.play_forward()
                    elif event.key == pygame.K_LEFT:
                        self.play_reverse()
                    elif event.key == pygame.K_s:
                        self.stop_playback()
                
                # Handle button clicks
                if self.play_btn.handle_event(event):
                    self.play_forward()
                if self.reverse_btn.handle_event(event):
                    self.play_reverse()
                if self.stop_btn.handle_event(event):
                    self.stop_playback()
                
                # Handle slider
                self.speed_slider.handle_event(event)
            
            # Playback logic
            if self.playback_state == 'forward':
                speed = self.speed_slider.speed
                step = max(1, int(speed * 2))  # Frames to advance per tick
                
                self.current_frame = min(self.current_frame + step, len(self.game.moments) - 1)
                
                # Track ball carrier during playback to detect passes
                moment = self.game.moments[self.current_frame]
                self.update_ball_carrier(moment)
                
                # Add ball position to trail with current possessing team
                team_id = None
                if self.current_ball_carrier_id and self.current_ball_carrier_id in self.game.players:
                    team_id = self.game.players[self.current_ball_carrier_id].team_id
                self.ball_trail.append((moment.ball_pos, team_id))
                
                # Check if we reached the next event
                if self.current_frame >= self.next_event_frame:
                    self.current_frame = self.next_event_frame
                    self.on_event_reached(self.current_event_index, 'forward')
                    
            elif self.playback_state == 'reverse':
                speed = self.speed_slider.speed
                step = max(1, int(speed * 2))
                
                self.current_frame = max(self.current_frame - step, 0)
                
                # Add ball position to trail (in reverse too for visual effect)
                moment = self.game.moments[self.current_frame]
                self.update_ball_carrier(moment)
                
                # Add with current possessing team
                team_id = None
                if self.current_ball_carrier_id and self.current_ball_carrier_id in self.game.players:
                    team_id = self.game.players[self.current_ball_carrier_id].team_id
                self.ball_trail.append((moment.ball_pos, team_id))
                
                # Check if we reached the previous event
                if self.current_frame <= self.prev_event_frame:
                    self.current_frame = self.prev_event_frame
                    self.on_event_reached(self.current_event_index, 'reverse')
            
            # Ensure frame is valid
            self.current_frame = max(0, min(self.current_frame, len(self.game.moments) - 1))
            moment = self.game.moments[self.current_frame]
            
            # Clear and draw court
            self.screen.fill(BG_COLOR)
            self.screen.blit(self.court_surface, (0, 0))
            
            # Draw legend
            self.draw_legend()
            
            # Draw players
            for player_id, (x, y) in moment.player_positions.items():
                self.draw_player(player_id, x, y)
            
            # Draw ball trail (comet effect) before the ball
            self.draw_ball_trail()
            
            # Draw ball
            bx, by, bz = moment.ball_pos
            self.draw_ball(bx, by, bz)
            
            # ===== ALWAYS DRAW TEAM HOOPS =====
            # Home team hoop (based on quarter)
            home_hoop = get_hoop_for_team(self.game.home_team_id, self.game.home_team_id, moment.quarter)
            away_hoop = get_hoop_for_team(self.game.away_team_id, self.game.home_team_id, moment.quarter)
            
            home_hoop_screen = court_to_screen(*home_hoop)
            away_hoop_screen = court_to_screen(*away_hoop)
            
            # Draw home team hoop (red)
            pygame.draw.circle(self.screen, HOME_COLOR, home_hoop_screen, 16, 3)
            # Draw away team hoop (blue)
            pygame.draw.circle(self.screen, AWAY_COLOR, away_hoop_screen, 16, 3)
            
            # ===== POSSESSION INDICATORS (always on during play) =====
            # Draw line from ball carrier to ball when someone has possession
            if self.current_ball_carrier_id and self.current_ball_carrier_pos:
                ball_xy = (moment.ball_pos[0], moment.ball_pos[1])
                carrier_screen = court_to_screen(*self.current_ball_carrier_pos)
                ball_screen = court_to_screen(*ball_xy)
                
                # Only draw if ball is close (possession)
                ball_dist = calculate_distance(self.current_ball_carrier_pos, ball_xy)
                if ball_dist < self.possession_threshold * 2:
                    # Get team color for the carrier
                    if self.current_ball_carrier_id in self.game.players:
                        carrier = self.game.players[self.current_ball_carrier_id]
                        carrier_color = HOME_COLOR if carrier.team_id == self.game.home_team_id else AWAY_COLOR
                        
                        # Draw possession indicator line
                        pygame.draw.line(self.screen, carrier_color, carrier_screen, ball_screen, 3)
                        
                        # Highlight the target hoop for the offensive team
                        target_hoop = get_hoop_for_team(carrier.team_id, self.game.home_team_id, moment.quarter)
                        target_hoop_screen = court_to_screen(*target_hoop)
                        
                        # Draw bright highlight ring on target hoop
                        pygame.draw.circle(self.screen, carrier_color, target_hoop_screen, 22, 5)
                        pygame.draw.circle(self.screen, (255, 255, 255), target_hoop_screen, 14, 2)
            
            # ===== STOPPED STATE OVERLAYS =====
            if self.current_analysis and self.playback_state == 'stopped':
                # Get team colors
                off_color, def_color = self.get_team_colors(self.current_analysis)
                
                # Offensive centroid - circle with crosshair (team color)
                if self.current_analysis.offensive_centroid != (0, 0):
                    cx, cy = court_to_screen(*self.current_analysis.offensive_centroid)
                    pygame.draw.circle(self.screen, off_color, (cx, cy), 12, 3)
                    pygame.draw.line(self.screen, off_color, (cx - 8, cy), (cx + 8, cy), 3)
                    pygame.draw.line(self.screen, off_color, (cx, cy - 8), (cx, cy + 8), 3)
                
                # Defensive centroid - square with X (team color)
                if self.current_analysis.defensive_centroid != (0, 0):
                    dx, dy = court_to_screen(*self.current_analysis.defensive_centroid)
                    pygame.draw.rect(self.screen, def_color, (dx - 10, dy - 10, 20, 20), 3)
                    pygame.draw.line(self.screen, def_color, (dx - 7, dy - 7), (dx + 7, dy + 7), 3)
                    pygame.draw.line(self.screen, def_color, (dx + 7, dy - 7), (dx - 7, dy + 7), 3)
                
                # Draw vector to hoop from ball carrier (offensive team color)
                if self.current_analysis.ball_carrier_pos and self.current_analysis.target_hoop:
                    carrier_screen = court_to_screen(*self.current_analysis.ball_carrier_pos)
                    hoop_screen = court_to_screen(*self.current_analysis.target_hoop)
                    pygame.draw.line(self.screen, off_color, carrier_screen, hoop_screen, 2)
            
            
            # Draw UI
            self.draw_scoreboard(moment)
            self.draw_controls()
            self.draw_event_log()
            self.draw_shot_celebration()
            self.draw_pending_shot_info()
            self.draw_analysis_panel()
            
            # Update display
            pygame.display.flip()
            
            # Control frame rate
            self.clock.tick(TARGET_FPS)
        
        pygame.quit()


# ============================================================================
# Main Entry Point
# ============================================================================

def list_available_games() -> List[str]:
    """Find all available game IDs that have both svu and pbp files."""
    import glob
    
    svu_files = glob.glob(os.path.join(DATA_DIR, '*_svu.json'))
    pbp_files = glob.glob(os.path.join(DATA_DIR, '*_pbp.json'))
    
    svu_ids = {os.path.basename(f).replace('_svu.json', '') for f in svu_files}
    pbp_ids = {os.path.basename(f).replace('_pbp.json', '') for f in pbp_files}
    
    # Games that have both files
    complete_games = sorted(svu_ids & pbp_ids)
    return complete_games


def main():
    if len(sys.argv) > 1:
        game_id = sys.argv[1]
    else:
        game_id = '0021500001'  # Default game
    
    try:
        game_data = load_game(game_id)
        renderer = GameRenderer(game_data)
        renderer.run()
    except FileNotFoundError as e:
        print(f"Error: Could not find game files for {game_id}")
        print(f"Make sure {DATA_DIR}/{game_id}_svu.json and {DATA_DIR}/{game_id}_pbp.json exist.")
        print(f"Details: {e}")
        
        # List available games
        available = list_available_games()
        if available:
            print(f"\nAvailable games with complete data:")
            for gid in available:
                print(f"  - {gid}")
        else:
            print("\nNo complete game files found in current directory.")
            print(f"Each game needs both {DATA_DIR}/{{game_id}}_svu.json and {DATA_DIR}/{{game_id}}_pbp.json files.")
        
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
