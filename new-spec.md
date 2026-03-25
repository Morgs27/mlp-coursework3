
I did some preliminary testing in this codebase to check if it was possible to predict gaze direction from face landmarks and then use that to predict where a dart player is aiming on the board using game replays broadcasts. 

This was sucessful. A set of /debug-images annotated to /annotated-images were used and the output can be shown in the gaze_plots. 

The next step is to take this proof on concept and merge it in with a larger academic paper on using ML techniques to predict the next players throw. 

This will involve:
1. Build out a module that takes a frame and returns a set of gaze information as described bellow
2. Create a dataset of frames from right before a throw from a set of video matches. 
3. Pass this dataset through the module 1. to produce a dataset of gaze vectors to score etc as described in the labeling bellow
4. Train a basic linear regression model to predict the resulting score from the gaze vectors
5. Run some basic evaluation on the model & the dataset. Keep some of the original dataset for testing.
6. Build a proof on concept web UI. This will in real time play a video, annotate the frame with the gaze vectors, and use the trained model to predict the resulting score from the gaze vectors. 
7. Finally to train a deep network later on, I will need a vector encoder to pass into a transformer reranker. 

Number 1 has somewhat been implemented in the gaze_tracking.py file. Just make it a clean and usable module. 
Number 2 will be very difficult & might have to be done manually. The feeds from videos are patchy with different views etc. We need to think of some smart way of doing this. 
Number 3 will be very easy once number 2 is done. 
Number 4 should be straightforward once we have data. 
Number 5 should produce graphs in a format for an academic paper. PDF & Reasonable colours and labels etc.
Number 6 should be a very simple UI that imports some basic UI library. 
Number 7 is not too important for now. 

Each section should be very clean code as the code will be published and in it's own folder. Don't use the code currently in place. 

# Details

Labeling the frame dataset. If the throw can't be matched all fields will have to be manual. Otherwise the player_id & resulting_score can be extracted based on the timestamp?

throw_id
player_id
frame_path
resulting_score
match_id

Gaze output

left gaze vector (x,y,z)
right gaze vector (x,y,z)
averaged gaze vector (x,y,z)
head axes / pose frame
IPD
maybe left-right eye agreement
maybe detector confidence / valid-face flag

# Data collection

I'll manually add videos in the /video folder. 

An api of darts matches with scores and player can be found by using the following code:

import time, random, requests
import pandas as pd

API_KEY = "YOUR_API_KEY_HERE"
ACCESS_LEVEL = "trial"     # "trial" or "production"
LANG = "en"
FORMAT = "json"

def sr_get(url, params, timeout=30, max_retries=8, base_sleep=1.0):
    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=timeout)

        if r.status_code == 200:
            return r

        if r.status_code in (429, 500, 502, 503, 504):
            retry_after = r.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after else base_sleep * (2 ** attempt)
            sleep_s += random.uniform(0, 0.5)
            print(f"HTTP {r.status_code} → retry {attempt+1}/{max_retries} in {sleep_s:.2f}s")
            time.sleep(sleep_s)
            continue

        r.raise_for_status()

    raise RuntimeError(f"Failed after {max_retries} retries: {url}")

# 1) Get live match list
summ_url = f"https://api.sportradar.com/darts/{ACCESS_LEVEL}/v2/{LANG}/schedules/live/summaries.{FORMAT}"
live = sr_get(summ_url, params={"api_key": API_KEY}).json()

events = live.get("sport_events") or live.get("summaries") or []
if not events:
    raise RuntimeError("No live events returned. Try daily summaries instead.")

# pick first event
se = events[0].get("sport_event", events[0])
SPORT_EVENT_ID = se["id"]

print("Using match:", SPORT_EVENT_ID,
      "| competition:", se.get("sport_event_context", {}).get("competition", {}).get("name"),
      "| start:", se.get("start_time"))

# 2) Fetch match timeline (dart-by-dart)
tl_url = f"https://api.sportradar.com/darts/{ACCESS_LEVEL}/v2/{LANG}/sport_events/{SPORT_EVENT_ID}/timeline.{FORMAT}"
timeline = sr_get(tl_url, params={"api_key": API_KEY}).json()

# 3) Extract dart events
def seg(score, mult):
    if score is None or mult is None:
        return None
    if score == 25 and mult == 1: return "SB"
    if score == 25 and mult == 2: return "DB"
    if mult == 1: return f"S{score}"
    if mult == 2: return f"D{score}"
    if mult == 3: return f"T{score}"
    return f"{mult}x{score}"

rows = []
for ev in timeline.get("timeline", []):
    if ev.get("type") == "dart":
        score = ev.get("dart_score")
        mult  = ev.get("dart_score_multiplier")
        total = ev.get("dart_score_total")
        rows.append({
            "event_id": ev.get("id"),
            "time": ev.get("time"),
            "competitor": ev.get("competitor"),  # home/away
            "dart_score": score,
            "multiplier": mult,
            "dart_total": total,
            "segment": seg(score, mult),
            "is_bust": ev.get("is_bust"),
            "is_checkout_attempt": ev.get("is_checkout_attempt"),
            "is_gameshot": ev.get("is_gameshot"),
            "home_score": ev.get("home_score"),
            "away_score": ev.get("away_score"),
            "period": ev.get("period"),
        })

df_darts = pd.DataFrame(rows).sort_values(["time", "event_id"])
df_darts.head(200)
