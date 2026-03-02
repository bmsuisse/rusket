"""
Time-Aware Sequential Recommendation Example

This script demonstrates how to use the `time_aware=True` feature in sequential 
models like SASRec and FPMC. 

By default, sequential models only care about the *order* of items (Item A -> Item B). 
When `time_aware=True` is enabled, the model will additionally learn an embedding 
for the actual physical time elapsed (in days) between user interactions. This helps 
the model understand that an action taken 1 second ago has a different context than 
an action taken 300 days ago!
"""

import pandas as pd
from rusket import SASRec, FPMC

def main():
    print("🚀 Preparing Time-Aware Event Data...")
    
    # Let's mock a scenario with two users.
    # Both have the exact same sequence of item views, but User 1 did them all in one day,
    # and User 2 did them spread out over a year.
    events = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2],
        "item_id": [10, 20, 30, 10, 20, 30],
        # Timestamps are in unix epochs.
        "timestamp": [
            1000, 1005, 1010, # User 1: fast clicks (seconds apart)
            1000, 1000 + 86400 * 100, 1000 + 86400 * 200 # User 2: months apart
        ]
    })
    
    print("\n--- SASRec (Self-Attentive Sequential Recommendation) ---")
    # Initialize the SASRec model with time_aware=True
    sasrec = SASRec(
        factors=16,
        n_layers=2,
        time_aware=True,
        max_time_steps=256, # The maximum number of days difference to track
        iterations=100
    )
    
    # Fit the model, passing the timestamp column
    sasrec = sasrec.from_transactions(
        events, 
        user_col="user_id", 
        item_col="item_id", 
        timestamp_col="timestamp"
    )
    sasrec.fit()
    
    # Predict the next item for an anonymous session
    session = [10, 20] # They viewed 10, then 20
    
    # If they viewed them right now (timestamps 10 and 20):
    fast_items, fast_scores = sasrec.recommend_items(session, timestamps=[10, 20], n=3)
    print(f"Prediction for fast session: {fast_items} (Scores: {fast_scores})")
    
    # If they viewed them months apart:
    slow_items, slow_scores = sasrec.recommend_items(session, timestamps=[10, 10 + 86400*100], n=3)
    print(f"Prediction for slow session: {slow_items} (Scores: {slow_scores})")
    
    
    print("\n--- FPMC (Factorized Personalized Markov Chains) ---")
    # Initialize the FPMC model with time_aware=True
    fpmc = FPMC(
        factors=16,
        time_aware=True,
        max_time_steps=256,
        iterations=100
    )
    
    # Fit FPMC
    fpmc = fpmc.from_transactions(
        events, 
        user_col="user_id", 
        item_col="item_id", 
        timestamp_col="timestamp"
    )
    fpmc.fit()
    
    # FPMC makes predictions per unique user based on their last known interaction
    u1, u2 = 1, 2 
    
    # Predict next item for User 1 if they interacted "today"
    # Note: timestamps for predictions here are the target time for the generic Next Item
    item1, score1 = fpmc.recommend_items(user_id=0, timestamp=1011, n=3) # user 0 mapped internally for 1
    
    # Predict next item for User 1 if they interacted "next year"
    item2, score2 = fpmc.recommend_items(user_id=0, timestamp=1011 + 86400*100, n=3)
    
    print(f"FPMC Prediction if bought immediately: {item1} (Scores: {score1})")
    print(f"FPMC Prediction if bought months later: {item2} (Scores: {score2})")

if __name__ == "__main__":
    main()
