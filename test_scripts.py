import engine
from pathlib import Path

# 1. Pick a photo of yourself that IS NOT in the 89 photos
query_image = "me.jpg" 

if not Path(query_image).exists():
    print(f"❌ Error: Please place a photo named '{query_image}' in the folder.")
else:
    print(f"🔍 Searching for matches to {query_image}...")
    
    # 2. Call our engine to find the top 3 matches
    # This triggers the vector math we discussed earlier
    results = engine.search_by_face(query_image, top_k=10)
    
    if results:
        print(f"✅ Found {len(results)} matches!")
        for i, path in enumerate(results):
            print(f"Match {i+1}: {path}")
    else:
        print("❓ No matches found. Try checking if the face is clear in the photo.")