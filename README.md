
<div align="center">

# **𝑮𝒓𝒐𝒖𝒑 𝟏𝟏 𝑭𝒊𝒏𝒂𝒍: 𝑭𝒂𝒄𝒆 𝑹𝒆𝒄𝒐𝒈𝒏𝒊𝒕𝒊𝒐𝒏**


</div>



𝑨𝒖𝒕𝒉𝒐𝒓/𝒔: Justin Joe Arellano, Ralph Nathanael Dela Peña, 𝘢𝘯𝘥 Lenard Albert Fajardo

<div align="justify">
    
&nbsp;&nbsp;&nbsp;&nbsp;Levi Strauss & Co. is one of the world's largest brand-name apparel companies and a global leader in jeanswear. The company designs and markets jeans, casual wear and related accessories for men, women and children under the Levi's®, Dockers®, Signature by Levi Strauss & Co.™, and Denizen® brands. Its products are sold in more than 110 countries worldwide through a combination of chain retailers, department stores, online sites, and a global footprint of approximately 3,000 retail stores and shop-in-shops.

</div>

<div align="center">

---

### 𝐃𝐞𝐧𝐢𝐦 𝐅𝐮𝐬𝐢𝐨𝐧: 𝐋𝐞𝐯𝐢'𝐬 𝐱 𝐍𝐞𝐰𝐉𝐞𝐚𝐧𝐬 𝐔𝐧𝐯𝐞𝐢𝐥𝐬 𝐚 𝐓𝐢𝐦𝐞𝐥𝐞𝐬𝐬 𝐏𝐚𝐫𝐭𝐧𝐞𝐫𝐬𝐡𝐢𝐩

---

</div>

<p align="center">
  <img width="600" height="800" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/blob/5b1b89f0cefa3af52506a258b1d01fad064ecef8/GP3.jpg">
</p>

<div align="center">

### _NewJeans (뉴진스) is a 5-member girl group under ADOR and HYBE Labels. The members consist of Minji, Hanni, Danielle, Haerin, and Hyein. They released their debut single “Attention” on July 22, 2022, followed by their debut extended play, New Jeans, which was released on August 1, 2022._
</div>

<div align="justify">

&nbsp;&nbsp;&nbsp;&nbsp;Levi's is partnering with K-pop girl group NewJeans to engage younger customers. NewJeans gained fame with hits like "Attention" and made it to the Billboard Hot 100. The collaboration signifies the group's aspiration to be as timeless as Levi's iconic 501 jeans. The Levi's Spring/Summer 2023 collection will feature a marketing campaign showcasing NewJeans' positive attitude. The group will wear Levi's 501 '81 jeans and 501 Original jeans in promotional materials. NewJeans will perform in Seoul on May 20th to celebrate 501 Day. Levi's sees the collaboration as a way to connect with NewJeans' global fanbase in an authentic manner._

</div>

>
> 

<div align="center">

---

### 𝐅𝐀𝐂𝐄 𝐑𝐄𝐂𝐎𝐆𝐍𝐈𝐓𝐈𝐎𝐍 𝐂𝐎𝐃𝐄𝐒

---

</div>

### 📋 𝐂𝐨𝐝𝐞: 𝐈𝐦𝐩𝐨𝐫𝐭𝐢𝐧𝐠 𝐨𝐟 𝐭𝐡𝐞 𝐈𝐦𝐚𝐠𝐞𝐬 𝐟𝐫𝐨𝐦 𝐭𝐡𝐞 𝐆𝐢𝐭𝐡𝐮𝐛 𝐚𝐧𝐝 𝐈𝐧𝐬𝐭𝐚𝐥𝐥𝐢𝐧𝐠 𝐭𝐡𝐞 "𝐟𝐚𝐜𝐞_𝐫𝐞𝐜𝐨𝐠𝐧𝐢𝐭𝐢𝐨𝐧"
    !git clone https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition.git
    !pip install face_recognition
    %cd Group11_Finals_FaceRecognition

### 📋 𝐂𝐨𝐝𝐞: 𝐄𝐧𝐜𝐨𝐝𝐢𝐧𝐠 𝐏𝐫𝐨𝐟𝐢𝐥𝐞𝐬 𝐔𝐬𝐢𝐧𝐠 𝐊𝐧𝐨𝐰𝐧 𝐅𝐚𝐜𝐞 𝐈𝐦𝐚𝐠𝐞𝐬
    import face_recognition
    import numpy as np
    from google.colab.patches import cv2_imshow
    import cv2
    
    # Creating the encoding profiles
    face_1 = face_recognition.load_image_file("Danielle.jpeg")
    face_1_encoding = face_recognition.face_encodings(face_1)[0]
    
    face_2 = face_recognition.load_image_file("Haerin.jpg")
    face_2_encoding = face_recognition.face_encodings(face_2)[0]
    
    face_3 = face_recognition.load_image_file("Haerin.jpg")
    face_3_encoding = face_recognition.face_encodings(face_3)[0]
    
    face_4 = face_recognition.load_image_file("Hyein.jpeg")
    face_4_encoding = face_recognition.face_encodings(face_4)[0]
    
    face_5 = face_recognition.load_image_file("Minji.jpeg")
    face_5_encoding = face_recognition.face_encodings(face_5)[0]
    
    known_face_encodings = [
                            face_1_encoding,
                            face_2_encoding,
                            face_3_encoding,
                            face_4_encoding,
                            face_5_encoding
    ]
    
    known_face_names = [
                        "Danielle",
                        "Haerin",
                        "Hanni",
                        "Hyein",
                        "Minji",
    ]

### 📋 𝐂𝐨𝐝𝐞: 𝐑𝐮𝐧𝐧𝐢𝐧𝐠 𝐭𝐡𝐞 𝐅𝐚𝐜𝐞 𝐑𝐞𝐜𝐨𝐠𝐧𝐢𝐭𝐢𝐨𝐧 𝐨𝐧 𝐭𝐡𝐞 𝐀𝐦𝐛𝐚𝐬𝐬𝐚𝐝𝐫𝐞𝐬𝐬 𝐨𝐟 𝐋𝐞𝐯𝐢𝐬

<div align="justify">

&nbsp;&nbsp;&nbsp;&nbsp;𝖯𝗋𝗈𝗏𝗂𝖽𝖾𝖽 𝖻𝖾𝗅𝗈𝗐 𝗂𝗌 𝖺 𝖯𝗒𝗍𝗁𝗈𝗇 𝖼𝗈𝖽𝖾 𝗍𝗁𝖺𝗍 𝗉𝖾𝗋𝖿𝗈𝗋𝗆𝗌 𝖿𝖺𝖼𝖾 𝗋𝖾𝖼𝗈𝗀𝗇𝗂𝗍𝗂𝗈𝗇 𝗈𝗇 𝖺𝗇 𝗂𝗆𝖺𝗀𝖾 𝗎𝗌𝗂𝗇𝗀 𝗍𝗁𝖾 𝖿𝖺𝖼𝖾_𝗋𝖾𝖼𝗈𝗀𝗇𝗂𝗍𝗂𝗈𝗇 𝗅𝗂𝖻𝗋𝖺𝗋𝗒 𝖺𝗇𝖽 𝖮𝗉𝖾𝗇𝖢𝖵. 𝖨𝗍 𝗅𝗈𝖺𝖽𝗌 𝖺𝗇 𝗎𝗇𝗄𝗇𝗈𝗐𝗇 𝗂𝗆𝖺𝗀𝖾, 𝖽𝖾𝗍𝖾𝖼𝗍𝗌 𝖿𝖺𝖼𝖾𝗌, 𝖺𝗇𝖽 𝖼𝗈𝗆𝗉𝖺𝗋𝖾𝗌 𝗍𝗁𝖾𝗂𝗋 𝖾𝗇𝖼𝗈𝖽𝗂𝗇𝗀𝗌 𝗐𝗂𝗍𝗁 𝖺 𝗌𝖾𝗍 𝗈𝖿 𝗄𝗇𝗈𝗐𝗇 𝖿𝖺𝖼𝖾 𝖾𝗇𝖼𝗈𝖽𝗂𝗇𝗀𝗌. 𝖳𝗁𝖾 𝖼𝗈𝖽𝖾 𝗍𝗁𝖾𝗇 𝖽𝗋𝖺𝗐𝗌 𝗋𝖾𝖼𝗍𝖺𝗇𝗀𝗅𝖾𝗌 𝖺𝗋𝗈𝗎𝗇𝖽 𝗋𝖾𝖼𝗈𝗀𝗇𝗂𝗓𝖾𝖽 𝖿𝖺𝖼𝖾𝗌, 𝖺𝗇𝗇𝗈𝗍𝖺𝗍𝖾𝗌 𝗍𝗁𝖾𝗆 𝗐𝗂𝗍𝗁 𝖼𝗈𝗋𝗋𝖾𝗌𝗉𝗈𝗇𝖽𝗂𝗇𝗀 𝗇𝖺𝗆𝖾𝗌, 𝖺𝗇𝖽 𝖽𝗂𝗌𝗉𝗅𝖺𝗒𝗌 𝗍𝗁𝖾 𝗆𝗈𝖽𝗂𝖿𝗂𝖾𝖽 𝗂𝗆𝖺𝗀𝖾, 𝗌𝗁𝗈𝗐𝖼𝖺𝗌𝗂𝗇𝗀 𝗍𝗁𝖾 𝗋𝖾𝗌𝗎𝗅𝗍𝗌 𝗈𝖿 𝗍𝗁𝖾 𝖿𝖺𝖼𝖾 𝗋𝖾𝖼𝗈𝗀𝗇𝗂𝗍𝗂𝗈𝗇 𝗉𝗋𝗈𝖼𝖾𝗌𝗌.
>
✍️To use the code below you must encode the file name of the picture to the _**[file_name = " "]**_
>
📌𝖥𝗈𝗋 𝗍𝗁𝖾 𝖿𝖺𝖼𝖾 𝗋𝖾𝖼𝗈𝗀𝗇𝗂𝗍𝗂𝗈𝗇 𝗈𝖿 𝗍𝗁𝖾 𝖺𝗆𝖻𝖺𝗌𝗌𝖺𝖽ress 𝗈𝖿 Levis 𝗍𝗁𝖾 𝗀𝗋𝗈𝗎𝗉 𝗎𝗍𝗂𝗅𝗂𝗓𝖾𝖽 the code below 𝖿𝗈𝗋 both 𝐊𝐧𝐨𝐰𝐧 𝖺𝗇𝖽 𝐔𝐧𝐤𝐧𝐨𝐰𝐧 𝗂𝖽𝖾𝗇𝗍𝗂𝗍𝗂𝖾𝗌:
> </div>

        file_name = " "
        unknown_image = face_recognition.load_image_file(file_name)
        unknown_image_to_draw = cv2.imread(file_name)
        
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        
        for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
          matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
          name = "Unknown"
        
          face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
          best_match_index = np.argmin(face_distances)
          if matches[best_match_index]:
            name = known_face_names[best_match_index]
          cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
          cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
        
        cv2_imshow(unknown_image_to_draw)
>
>         
<div align="center">

---

### 𝐍𝐞𝐰𝐉𝐞𝐚𝐧𝐬 𝐗 𝐅𝐚𝐜𝐞 𝐑𝐞𝐜𝐨𝐠𝐧𝐢𝐭𝐢𝐨𝐧

---

</div>


<p align="center">
  <img width="800" height="469" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/01359a7f-a6ef-42b5-bfcb-a0a6288b6fb7">
</p>

<div align="justify">
𝖥𝖺𝖼𝖾 𝗋𝖾𝖼𝗈𝗀𝗇𝗂𝗍𝗂𝗈𝗇 𝗂𝗌 𝖺 𝖻𝗂𝗈𝗆𝖾𝗍𝗋𝗂𝖼 𝗍𝖾𝖼𝗁𝗇𝗈𝗅𝗈𝗀𝗒 𝗍𝗁𝖺𝗍 𝗂𝖽𝖾𝗇𝗍𝗂𝖿𝗂𝖾𝗌 𝗈𝗋 𝗏𝖾𝗋𝗂𝖿𝗂𝖾𝗌 𝗂𝗇𝖽𝗂𝗏𝗂𝖽𝗎𝖺𝗅𝗌 𝖻𝗒 𝖺𝗇𝖺𝗅𝗒𝗓𝗂𝗇𝗀 𝖺𝗇𝖽 𝗆𝖺𝗍𝖼𝗁𝗂𝗇𝗀 𝗍𝗁𝖾𝗂𝗋 𝖿𝖺𝖼𝗂𝖺𝗅 𝖿𝖾𝖺𝗍𝗎𝗋𝖾𝗌. 𝖳𝗁𝖾 𝗉𝗋𝗈𝖼𝖾𝗌𝗌 𝗂𝗇𝗏𝗈𝗅𝗏𝖾𝗌 𝖿𝖺𝖼𝖾 𝖽𝖾𝗍𝖾𝖼𝗍𝗂𝗈𝗇, 𝖼𝖺𝗉𝗍𝗎𝗋𝗂𝗇𝗀 𝖿𝖺𝖼𝗂𝖺𝗅 𝗂𝗆𝖺𝗀𝖾𝗌, 𝖾𝗑𝗍𝗋𝖺𝖼𝗍𝗂𝗇𝗀 𝖽𝗂𝗌𝗍𝗂𝗇𝖼𝗍𝗂𝗏𝖾 𝖿𝖾𝖺𝗍𝗎𝗋𝖾𝗌, 𝖺𝗇𝖽 𝖼𝗈𝗆𝗉𝖺𝗋𝗂𝗇𝗀 𝗍𝗁𝖾𝗆 𝗐𝗂𝗍𝗁 𝗉𝗋𝖾-𝗌𝗍𝗈𝗋𝖾𝖽 𝖽𝖺𝗍𝖺 𝗂𝗇 𝖺 𝖽𝖺𝗍𝖺𝖻𝖺𝗌𝖾. 𝖨𝗍 𝗂𝗌 𝗎𝗌𝖾𝖽 𝖿𝗈𝗋 𝗏𝖺𝗋𝗂𝗈𝗎𝗌 𝖺𝗉𝗉𝗅𝗂𝖼𝖺𝗍𝗂𝗈𝗇𝗌, 𝗂𝗇𝖼𝗅𝗎𝖽𝗂𝗇𝗀 𝗌𝖾𝖼𝗎𝗋𝗂𝗍𝗒, 𝖺𝖼𝖼𝖾𝗌𝗌 𝖼𝗈𝗇𝗍𝗋𝗈𝗅, 𝖺𝗇𝖽 𝗎𝗌𝖾𝗋 𝖺𝗎𝗍𝗁𝖾𝗇𝗍𝗂𝖼𝖺𝗍𝗂𝗈𝗇. 𝖶𝗁𝗂𝗅𝖾 𝖿𝖺𝖼𝖾 𝗋𝖾𝖼𝗈𝗀𝗇𝗂𝗍𝗂𝗈𝗇 𝗈𝖿𝖿𝖾𝗋𝗌 𝗇𝗈𝗇-𝗂𝗇𝗍𝗋𝗎𝗌𝗂𝗏𝖾 𝖺𝗇𝖽 𝖼𝗈𝗇𝗏𝖾𝗇𝗂𝖾𝗇𝗍 𝗌𝗈𝗅𝗎𝗍𝗂𝗈𝗇𝗌, 𝗉𝗋𝗂𝗏𝖺𝖼𝗒 𝖺𝗇𝖽 𝖽𝖺𝗍𝖺 𝗌𝖾𝖼𝗎𝗋𝗂𝗍𝗒 𝖼𝗈𝗇𝖼𝖾𝗋𝗇𝗌 𝗁𝖺𝗏𝖾 𝖻𝖾𝖾𝗇 𝗋𝖺𝗂𝗌𝖾𝖽 𝖽𝗎𝖾 𝗍𝗈 𝗂𝗍𝗌 𝗉𝗈𝗍𝖾𝗇𝗍𝗂𝖺𝗅 𝖿𝗈𝗋 𝗆𝗂𝗌𝗎𝗌𝖾 𝖺𝗇𝖽 𝗎𝗇𝖺𝗎𝗍𝗁𝗈𝗋𝗂𝗓𝖾𝖽 𝗌𝗎𝗋𝗏𝖾𝗂𝗅𝗅𝖺𝗇𝖼𝖾.

</div>

<div align="center">

---
### 𝐉𝐞𝐚𝐧𝐬 𝐓𝐡𝐚𝐭 𝐏𝐨𝐩**: 𝐋𝐞𝐯𝐢'𝐬 𝐚𝐧𝐝 𝐍𝐞𝐰𝐉𝐞𝐚𝐧𝐬 𝐌𝐚𝐤𝐞 𝐢𝐭 𝐒𝐢𝐦𝐩𝐥𝐞 𝐲𝐞𝐭 𝐒𝐭𝐲𝐥𝐢𝐬𝐡
---
### 🐰(𝐊𝐧𝐨𝐰𝐧 𝐱 𝐅𝐚𝐜𝐞 𝐑𝐞𝐜𝐨𝐠𝐧𝐢𝐭𝐢𝐨𝐧)🐰


</div>  

>
𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟏 (𝐃𝐀𝐍𝐈𝐄𝐋𝐋𝐄)
>
<div align="justify">
    
- Embrace the trendsetting vibes with Levi's® Women's Superlow Shorts paired effortlessly with the Graphic Cindy Long-Sleeve Top, as seen on the fabulous NewJeans Danielle. Elevate your fashion game with this iconic combo that exudes style and confidence
</div>

🐰 𝗖𝗼𝘀𝘁: PHP 2,199 (Top)
>
🐰 𝗖𝗼𝘀𝘁: PHP 2,499 (Bottom)

> >
 🥕 Utilizing the code for facial recognition and processing an image named _**"DanielleL.jpg"**_ produces the following outcome:

>
<p align="center">
  <img width="600" height="799" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/416ff9c1-84c1-411b-9755-5f0602606444">
</p>


𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟐 (𝐇𝐀𝐄𝐑𝐈𝐍)
>
<div align="justify">
    
- Channel the effortlessly cool vibe of NewJeans Haerin with Levi's® Women's Dry Goods V-Neck T-Shirt and Baggy Dad Jeans. Elevate your style game with this iconic duo, where comfort meets bold fashion in the most laid-back yet trendy way.

</div>

🐰 𝗖𝗼𝘀𝘁: PHP 1,699 (Top)
>
🐰 𝗖𝗼𝘀𝘁: PHP 3,999 (Bottom)

> >
🥕 Utilizing the code for facial recognition and processing an image named _**"HaerinL.jpg"**_ produces the following outcome: 

>
<p align="center">
  <img width="600" height="799" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/2f82ddef-b7d1-4a53-bccd-8dd95fea5c4e">
</p>


𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟑 (𝐇𝐀𝐍𝐍𝐈)
>
<div align="justify">
    
- Embrace the effortlessly cool style of NewJeans Hanni with Levi's® Women's Graphic Ringer Mini T-Shirt. This fashion-forward piece seamlessly blends comfort and trendiness, making it a must-have for your wardrobe.

</div>

🐰 𝗖𝗼𝘀𝘁: PHP 1,499 (Top)

> >
 🥕 Utilizing the code for facial recognition and processing an image named _**"HanniLL.jpg"**_ produces the following outcome:

>
<p align="center">
  <img width="600" height="1068" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/21bf9ba8-8bf3-42cb-bbaa-0ae06557073e">
</p>


𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟒 (𝐇𝐘𝐄𝐈𝐍)
>
<div align="justify">
    
- Elevate your street style with the Levi's® Women's Nola Oversized Shirt and Baggy Cargo Pants, as worn by the trendsetter NewJeans Hyein. Embrace comfort and fashion effortlessly in this iconic duo, reflecting Hyein's signature laid-back yet chic aesthetic

</div>

🐰 𝗖𝗼𝘀𝘁: PHP 2,999 (Top)
>
🐰 𝗖𝗼𝘀𝘁: PHP 2,499 (Bottom)

> >
 🥕 Utilizing the code for facial recognition and processing an image named _**"HyeinL.jpg**_ produces the following outcome: 

> 
<p align="center">
  <img width="600" height="799" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/6cfe2af8-9c20-45df-b776-4c3030830a7f">
</p>


𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝟓 (𝐌𝐈𝐍𝐉𝐈)
>
<div align="justify">
    
- Own the spotlight in Levi's® Women's Ribcage Wide-Leg Jeans, as showcased by the style maven NewJeans Minji. Embrace comfort and fashion in this iconic pair, making a bold statement with every step.

</div>

🐰 𝗖𝗼𝘀𝘁: PHP 3,699 (Bottom)

> >
 🥕 Utilizing the code for facial recognition and processing an image named _**"MinjiL.jpg**_ produces the following outcome:  

>
<p align="center">
  <img width="600" height="799" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/a1a7dce1-bb50-418a-a747-ebd8e2baa76f">
</p>


<div align="center">

---    
### 𝐖𝐨𝐫𝐥𝐝𝐰𝐢𝐝𝐞 𝐅𝐚𝐬𝐡𝐢𝐨𝐧 𝐈𝐜𝐨𝐧𝐬.: 𝐀𝐦𝐛𝐚𝐬𝐬𝐚𝐝𝐫𝐞𝐬𝐬 𝐚𝐧𝐝 𝐌𝐨𝐝𝐞𝐥𝐬 𝐨𝐟 𝐋𝐞𝐯𝐢'𝐬 𝐂𝐥𝐨𝐭𝐡𝐢𝐧𝐠 𝐁𝐫𝐚𝐧𝐝
--- 
### 🐰(𝐔𝐧𝐤𝐧𝐨𝐰𝐧 𝐱 𝐅𝐚𝐜𝐞 𝐑𝐞𝐜𝐨𝐠𝐧𝐢𝐭𝐢𝐨𝐧)🐰
   
</div>  

<div align="justify">

Several international stars, renowned for their influence and style, represent Levi's Jeans as ambassadors and models. These individuals play a key role in promoting the brand's diverse and fashionable apparel. From the global stage to local markets, they embody the Levi's ethos, showcasing its commitment to inclusivity and contemporary fashion trends. Their collaborations with the brand contribute to its worldwide appeal and reflect the diversity of Levi's customer base, solidifying their status as influential figures in the fashion industry while reinforcing the brand's connection with consumers worldwide.

</div>
 

</div>

 𝐌𝐨𝐝𝐞𝐥 𝟏
>
> > 
 🥕 Utilizing the code for facial recognition and processing an image named _**"U1.jpg"**_ produces the following outcome: 
>
<p align="center">
  <img width="600" height="900" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/3cae43c5-cdfe-4611-8879-49cf8dbab698">
</p>


 𝐌𝐨𝐝𝐞𝐥 𝟐
>
> >
 🥕 Utilizing the code for facial recognition and processing an image named _**"U2.jpg"**_ produces the following outcome: 
>
<p align="center">
  <img width="600" height="900" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/93e873c2-2b47-49ef-98ec-e0ffe271ef8d">
</p>


 𝐌𝐨𝐝𝐞𝐥 𝟑
>
> >
 🥕 Utilizing the code for facial recognition and processing an image named _**"U3.jpg"**_ produces the following outcome: 
>
<p align="center">
  <img width="600" height="888" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/5f28d1e3-04ec-4b54-b103-6f69ae648df5">
</p>


 𝐌𝐨𝐝𝐞𝐥 𝟒
>
> >
 🥕 Utilizing the code for facial recognition and processing an image named _**"U4.jpg"**_ produces the following outcome: 
> 
<p align="center">
  <img width="600" height="750" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/2625d872-ca23-4ffd-8c7f-ab324ba48099">
</p>


 𝐌𝐨𝐝𝐞𝐥 𝟓
>
> >
 🥕 Utilizing the code for facial recognition and processing an image named _**"U5.jpg"**_ produces the following outcome: 
>
<p align="center">
  <img width="408" height="544" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/f112ccb2-0128-4451-b3b0-d4e8334fb210">
</p>


 𝐌𝐨𝐝𝐞𝐥 𝟔
>
> >
 🥕 Utilizing the code for facial recognition and processing an image named _**"U6.jpg"**_ produces the following outcome: 
>
<p align="center">
  <img width="600" height="900" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/72638c26-5111-46e4-afad-b61f53cbd2f6">
</p>


 𝐌𝐨𝐝𝐞𝐥 𝟕
>
> >
 🥕 Utilizing the code for facial recognition and processing an image named _**"U7.jpg"**_ produces the following outcome: 
>
<p align="center">
  <img width="600" height="713" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/31085837-33b3-4cdf-b520-0bdae4ebc166">
</p>


 𝐌𝐨𝐝𝐞𝐥 𝟖
>
> > 
 🥕 Utilizing the code for facial recognition and processing an image named _**"U8.jpg"**_ produces the following outcome: 
>
<p align="center">
  <img width="600" height="770" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/6fc9f0b7-ca6c-4d31-bf85-5288abdcf2aa">
</p>


 𝐌𝐨𝐝𝐞𝐥 𝟗
>
> > 
 🥕 Utilizing the code for facial recognition and processing an image named _**"U9.jpg"**_ produces the following outcome: 
>
<p align="center">
  <img width="563" height="613" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/938b4a05-3cfd-4d30-8320-54df0fafbf08">
</p>


 𝐌𝐨𝐝𝐞𝐥 𝟏𝟎
>
> > 
 🥕 Utilizing the code for facial recognition and processing an image named _**"U10.jpg"**_ produces the following outcome: 
>
<p align="center">
  <img width="600" height="899" src="https://github.com/RalphNathanDP/Group11_Finals_FaceRecognition/assets/144073436/13756dc2-fb28-4d56-9a6f-790e0f4bcbb8">
</p>

___
## _𝐒𝐎𝐔𝐑𝐂𝐄𝐒 🌐:_

- https://twitter.com/newjeans_loop/status/1700728676680405255
- https://kpopping.com/kpics/230520-NewJeans-Danielle-LEVI-s-Event
- https://twitter.com/newjeans_loop/status/1700729048253821301
- https://kpopping.com/kpics/230520-NEWJEANS-Haerin-at-Levi-s-music-concert 
- https://www.pinterest.ph/pin/832954893607843554/
- https://twitter.com/newjeans_loop/status/1644165265289261056 
- https://www.pinterest.ph/pin/475974254383765922/ 
- https://kpopping.com/kpics/230520-NewJeans-Hyein-LEVI-s-Event 
- https://www.pinterest.ph/pin/773282198542729290/ 
- https://kpopping.com/kpics/230520-NewJeans-Minji-LEVI-s-Event 
- https://www.levistrauss.com/2023/03/14/k-pop-group-newjeans-partners-with-levis/ 
- https://kprofiles.com/newjeans-members-profile-facts/
- https://www.levi.com.ph/collections/women/newjeans 

___
