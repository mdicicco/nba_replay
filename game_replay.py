#!/usr/bin/env python3
"""
NBA Game Replay Visualization
Loads player tracking data and play-by-play events to create an animated 2D replay
of an NBA game using pygame with event-based navigation and adjustable speed.
"""

import json
import os
import pygame
import sys
from dataclasses import dataclass
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
WINDOW_WIDTH = int(COURT_LENGTH * COURT_SCALE)
WINDOW_HEIGHT = int(COURT_WIDTH * COURT_SCALE) + 250  # Extra space for controls

COURT_HEIGHT_PX = int(COURT_WIDTH * COURT_SCALE)

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
        
        # Draw current speed label
        speed_text = f"Speed: {self.speed}x"
        speed_surface = self.font_large.render(speed_text, True, SCORE_COLOR)
        screen.blit(speed_surface, (self.x + self.width + 20, self.y + self.height // 2 - 10))
    
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
    for i in range(0, WINDOW_WIDTH, 20):
        color = HARDWOOD_DARK if (i // 20) % 2 == 0 else HARDWOOD_LIGHT
        pygame.draw.rect(surface, color, (i, 0, 20, COURT_HEIGHT_PX))
    
    # Court boundaries
    pygame.draw.rect(surface, COURT_LINE_COLOR, (0, 0, WINDOW_WIDTH, COURT_HEIGHT_PX), 3)
    
    # Half court line
    half_x = WINDOW_WIDTH // 2
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
            base_x = WINDOW_WIDTH
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
            key_rect = pygame.Rect(WINDOW_WIDTH - key_length, key_top, key_length, key_width)
        
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
                           (WINDOW_WIDTH, corner_dist), 
                           (WINDOW_WIDTH - int(14 * COURT_SCALE), corner_dist), 3)
            pygame.draw.line(surface, THREE_POINT_COLOR,
                           (WINDOW_WIDTH, COURT_HEIGHT_PX - corner_dist),
                           (WINDOW_WIDTH - int(14 * COURT_SCALE), COURT_HEIGHT_PX - corner_dist), 3)
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
        
        # Pre-render court surface
        self.court_surface = pygame.Surface((WINDOW_WIDTH, COURT_HEIGHT_PX))
        draw_court(self.court_surface)
        
        # Game state
        self.current_frame = 0
        self.target_frame = 0
        self.running = True
        self.animating = False
        
        # Event navigation
        self.current_event_index = -1  # Start before first event
        
        # Score tracking
        self.home_score = 0
        self.away_score = 0
        
        # Event log (rolling text) - current + 4 past events
        self.event_log: deque = deque(maxlen=5)
        
        # Shot celebration display
        self.shot_display_text = ""
        self.shot_display_timer = 0
        
        # UI Components
        control_y = COURT_HEIGHT_PX + 60
        
        # Next Event button
        self.next_event_btn = Button(
            WINDOW_WIDTH // 2 - 80, control_y, 160, 40, "Next Event →"
        )
        
        # Previous Event button
        self.prev_event_btn = Button(
            WINDOW_WIDTH // 2 - 250, control_y, 160, 40, "← Prev Event"
        )
        
        # Speed slider
        self.speed_slider = SpeedSlider(
            50, control_y + 60, 350, 50
        )
        
        # Track processed events for score
        self.processed_events = set()
    
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
        
        # Handle shot made display
        if event.event_type == 1:  # Shot made
            player_name = event.player1_name or "Unknown"
            team_abbr = event.player1_team_abbr or ""
            self.shot_display_text = f"🏀 {player_name} ({team_abbr}) SCORES!"
            self.shot_display_timer = 90  # Display for ~1.5 seconds
    
    def go_to_next_event(self):
        """Navigate to the next event."""
        if self.current_event_index < len(self.game.events) - 1:
            self.current_event_index += 1
            event = self.game.events[self.current_event_index]
            self.target_frame = event.frame_index
            self.animating = True
            self.process_event(event)
    
    def go_to_prev_event(self):
        """Navigate to the previous event."""
        if self.current_event_index > 0:
            self.current_event_index -= 1
            event = self.game.events[self.current_event_index]
            self.target_frame = event.frame_index
            self.animating = True
            # Don't re-process events when going backward
    
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
        """Draw the basketball with height indication."""
        screen_pos = court_to_screen(x, y)
        
        # Ball size varies slightly with height for depth perception
        size = int(BALL_RADIUS + min(z * 0.5, 6))
        
        # Shadow
        shadow_offset = int(z * 0.3)
        pygame.draw.circle(self.screen, (50, 50, 50), 
                          (screen_pos[0] + 3, screen_pos[1] + 3 + shadow_offset), 
                          size - 2)
        
        # Ball
        pygame.draw.circle(self.screen, BALL_COLOR, screen_pos, size)
        pygame.draw.circle(self.screen, BALL_SHADOW, screen_pos, size, 2)
        
        # Ball seams for detail
        pygame.draw.arc(self.screen, (180, 80, 0), 
                       (screen_pos[0] - size, screen_pos[1] - size, size * 2, size * 2),
                       0.5, 2.5, 1)
    
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
                        (0, control_top, WINDOW_WIDTH, WINDOW_HEIGHT - control_top))
        
        # Draw buttons
        self.prev_event_btn.draw(self.screen)
        self.next_event_btn.draw(self.screen)
        
        # Draw slider
        self.speed_slider.draw(self.screen)
        
        # Animation status
        if self.animating:
            status = "▶ Playing..."
            status_color = (100, 255, 100)
        else:
            status = "⏸ Paused - Click 'Next Event' to advance"
            status_color = (255, 200, 100)
        
        status_text = self.font_small.render(status, True, status_color)
        self.screen.blit(status_text, (WINDOW_WIDTH // 2 + 100, COURT_HEIGHT_PX + 70))
    
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
    
    def draw_shot_celebration(self):
        """Draw shot made celebration overlay."""
        if self.shot_display_timer > 0:
            # Draw text with glow effect
            text = self.font_large.render(self.shot_display_text, True, SCORE_COLOR)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, COURT_HEIGHT_PX // 2))
            
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
    
    def run(self):
        """Main game loop."""
        print("Starting game replay...")
        print("Click 'Next Event' to advance through the game")
        print("Use the speed slider to adjust playback speed")
        print("Press Q to quit")
        
        while self.running and self.current_frame < len(self.game.moments):
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_SPACE:
                        self.go_to_next_event()
                    elif event.key == pygame.K_LEFT:
                        self.go_to_prev_event()
                
                # Handle button clicks
                if self.next_event_btn.handle_event(event):
                    self.go_to_next_event()
                if self.prev_event_btn.handle_event(event):
                    self.go_to_prev_event()
                
                # Handle slider
                self.speed_slider.handle_event(event)
            
            # Animation logic
            if self.animating:
                speed = self.speed_slider.speed
                
                if self.current_frame < self.target_frame:
                    # Moving forward
                    step = max(1, int(speed * 2))  # Frames to advance per tick
                    self.current_frame = min(self.current_frame + step, self.target_frame)
                elif self.current_frame > self.target_frame:
                    # Moving backward
                    step = max(1, int(speed * 2))
                    self.current_frame = max(self.current_frame - step, self.target_frame)
                else:
                    # Reached target
                    self.animating = False
            
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
            
            # Draw ball
            bx, by, bz = moment.ball_pos
            self.draw_ball(bx, by, bz)
            
            # Draw UI
            self.draw_scoreboard(moment)
            self.draw_controls()
            self.draw_event_log()
            self.draw_shot_celebration()
            
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
