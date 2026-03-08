# Take-home project: Image search & exploration experience

---

### **Objective**

Build a clothing image search and exploration experience inspired by [**same.energy**](http://same.energyhttps://same.energy/).

[Same Energy | Visual Search Engine](https://same.energy/)

## Mock

![Frame 157.jpg](attachment:c82e60f1-9956-4393-bc7d-b74074d89cf7:Frame_157.jpg)

### Key features

1. [Frontend] Garment highlight Animation (based on segmentation masks)
    1. You need to implement the exact animation, interactrion of Pinterest as much as possible
        1. hoaver animation
        2. click animation
        3. color and style
    
    ![https://www.pinterest.com/pin/86553624080468705/visual-search/?x=0&y=0&w=520&h=779.0780141843971&cropSource=5&rs=feed_home](attachment:efdc4eef-afd9-49a8-902e-7424e8bdae22:image.png)
    
    https://www.pinterest.com/pin/86553624080468705/visual-search/?x=0&y=0&w=520&h=779.0780141843971&cropSource=5&rs=feed_home
    
2. [Frontend] Optimization
    
    ![Frame 157.jpg](attachment:c82e60f1-9956-4393-bc7d-b74074d89cf7:Frame_157.jpg)
    
3. [Backend] Database & API

### **Core functionality**

- [ ]  Implement clothing ****based image search.
    - [ ]  Segmentation based garment highlight animation
        - Reference
            
            ![https://www.pinterest.com/pin/86553624080468705/visual-search/?x=0&y=0&w=520&h=779.0780141843971&cropSource=5&rs=feed_home](attachment:b638addb-fe7c-467c-8e1a-3b0d0a20ac48:image.png)
            
            https://www.pinterest.com/pin/86553624080468705/visual-search/?x=0&y=0&w=520&h=779.0780141843971&cropSource=5&rs=feed_home
            
    - [ ]  Clicking image would
- Display 1000+ images on screen smoothly, fast.
    - https://same.energy/ zoomed 25%, still works fast
        - curl
            
            ```bash
            curl 'https://imageapi.same.energy/search?i=0Gq6&nsfw=1&n=300' \
              -H 'Accept: */*' \
              -H 'Accept-Language: en-US,en;q=0.9' \
              -H 'Accept-Time: 27340953228847364729' \
              -H 'Connection: keep-alive' \
              -H 'Origin: https://same.energy' \
              -H 'Referer: https://same.energy/' \
              -H 'Sec-Fetch-Dest: empty' \
              -H 'Sec-Fetch-Mode: cors' \
              -H 'Sec-Fetch-Site: same-site' \
              -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36' \
              -H 'sec-ch-ua: "Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"' \
              -H 'sec-ch-ua-mobile: ?0' \
              -H 'sec-ch-ua-platform: "macOS"'
            ```
            
        
        ![image.png](attachment:f54c3190-b609-4048-bda5-a0efa96070e9:image.png)
        
        ![image.png](attachment:4b80b4d0-52ab-4b5f-ba9e-2d2e14815e03:image.png)
        
- Provide a navigation flow similar to same.energy, including:
    - Browser-native navigation (forward/back).
    - Every search result is shareable via a unique URL https://same.energy/search?i=xYAE

### What to build

- Frontend implementation
    - Optimal rendering of large number of images
    - Example apis
        
        ```jsx
        curl 'https://imageapi.same.energy/homepage' \
          -H 'Accept: */*' \
          -H 'Accept-Language: en-US,en;q=0.9' \
          -H 'Accept-Time: 08495924404339979522' \
          -H 'Connection: keep-alive' \
          -H 'Content-Type: text/plain;charset=UTF-8' \
          -H 'Origin: https://same.energy' \
          -H 'Sec-Fetch-Dest: empty' \
          -H 'Sec-Fetch-Mode: cors' \
          -H 'Sec-Fetch-Site: same-site' \
          -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36' \
          -H 'sec-ch-ua: "Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"' \
          -H 'sec-ch-ua-mobile: ?0' \
          -H 'sec-ch-ua-platform: "macOS"' \
          --data-raw '{"user_id":"anonymous@@10hztkdc99khvryk","token":""}'
          
          curl 'https://imageapi.same.energy/search?i=qxA5&nsfw=1&n=100' \
          -H 'Accept: */*' \
          -H 'Accept-Language: en-US,en;q=0.9' \
          -H 'Accept-Time: 55591410225377292695' \
          -H 'Connection: keep-alive' \
          -H 'Origin: https://same.energy' \
          -H 'Sec-Fetch-Dest: empty' \
          -H 'Sec-Fetch-Mode: cors' \
          -H 'Sec-Fetch-Site: same-site' \
          -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36' \
          -H 'sec-ch-ua: "Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"' \
          -H 'sec-ch-ua-mobile: ?0' \
          -H 'sec-ch-ua-platform: "macOS"'
        ```
        
- Backend implementation
    - Vector search
        - Database: use free tier https://qdrant.tech/pricing/ (**1GB free)**
        - Embedding: use free tier [**`*gemini-embedding-001*`**](https://ai.google.dev/gemini-api/docs/pricing#gemini-embedding)
    - Build database with [**Fashionpedia dataset](https://fashionpedia.github.io/home/Fashionpedia_download.html).**

### **Resources given**

- Access to an image dataset (100K+ items) with clothing-related tags.
    - Download script
        
        ```jsx
        #!/bin/bash
        
        set -e
        
        BASE_DIR="fashionpedia"
        mkdir -p "$BASE_DIR/images" "$BASE_DIR/annotations"
        
        cd "$BASE_DIR"
        
        download() {
          url="$1"
          out="$2"
          mkdir -p "$(dirname "$out")"
          echo "Downloading $out ..."
          wget -c -O "$out" "$url" &
        }
        
        # Images
        download "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip" \
                 "images/train2020.zip"
        
        download "https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip" \
                 "images/val_test2020.zip"
        
        # Detection annotations
        download "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json" \
                 "annotations/instances_attributes_train2020.json"
        
        download "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json" \
                 "annotations/instances_attributes_val2020.json"
        
        download "https://s3.amazonaws.com/ifashionist-dataset/annotations/info_test2020.json" \
                 "annotations/info_test2020.json"
        
        # Global attribute annotations
        download "https://s3.amazonaws.com/ifashionist-dataset/annotations/attributes_train2020.json" \
                 "annotations/attributes_train2020.json"
        
        download "https://s3.amazonaws.com/ifashionist-dataset/annotations/attributes_val2020.json" \
                 "annotations/attributes_val2020.json"
        
        # Wait for all background jobs to finish
        wait
        
        echo "All downloads finished."
        ```
        

### Other considerations

- Think about the user experience for someone who typically dresses basic, but is using this to find more unique clothes and brands. What would make this a good experience?
- Bonus points for high quality design engineering and demonstration of sharp taste.