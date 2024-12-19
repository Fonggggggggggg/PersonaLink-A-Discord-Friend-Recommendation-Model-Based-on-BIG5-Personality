# PersonaLink: A-Discord-Friend-Recommendation-Model-Based-on-BIG5-Personality
This is the final project of City University of Hong Kong Gifted Education Fund: Generative AI and AIoT (GenAIoT) Coding Skills Education for Gifted Students. Please note that this bot is not publically available and should be intended for personal use and entertainment purposes only.

## Descriptions
In this project, we enabled users to discover their personality traits through natural language, and then connects them with others who share similar characteristics within the same server.
We utilized the pre-trained model by Rong Wang, Kun Sun to assess user's BIG5 personality. (For more details, visit: <https://huggingface.co/KevSun/Personality_LM>) Then, we used euclidean distance to link users to the closest the other user who has the closest personality.

## Commands
The bot will record every message you sent in the server. By typing
### result : it will generate your BIG5 personality traits

### suggestion : it will recommed the user with the closest personality

### delete : it will delete all your past chat history saved inside the bot

## Limitations
1. Since the bot is not deployed onto the public server, the text file (conversation records) will only be stored on the local device
2. The bot can only recommend one person. This function can be expanded in future updates to provide multiple recommendations.
3. We originally planned to have a LLM chatbot to make the conversation more engaging. However, due to time limitations, we faild to put that into action.
