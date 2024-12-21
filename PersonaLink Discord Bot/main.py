import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from PredictionModel import TraitsResult
import pandas as pd
import numpy as np
import warnings
import torch
import os
import csv

###############Prediction model settings##########################
#It is normal that error will still appear, just ignore it
warnings.filterwarnings('ignore')
model = AutoModelForSequenceClassification.from_pretrained("KevSun/Personality_LM", ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained("KevSun/Personality_LM")
##################################################################

#User input will be stored and updated in this file
txt_file_path = ''

#if you created a .env file, use this
load_dotenv()
token = os.getenv('token')

#or you can insert your token here
#token = YOUR_GENERATED_TOKEN


#This ID is used in my dev server, please change it to your channel id
GUILD_ID = discord.Object(id=1315629249419284514)

#reading headers from csv file
headers = []
with open('Userdata.csv', mode='r') as file:
    reader = csv.reader(file)
    headers = next(reader)

class Client(commands.Bot):
    async def on_ready(self): 
        #check if the bot is activated, the message will be sent in the terminal
        print(f'Loggend on as {self.user}')

        #check if slash command is synced into guild
        try:
            GUILD_ID = discord.Object(id=1315629249419284514)
            synced = await self.tree.sync(guild=GUILD_ID)
            print(f'Synced {len(synced)} commands into guild {GUILD_ID}')

        except Exception as e:
            print(f'Error syncing commands: {e}')

    async def on_message(self, message):
    #check if the message is sent by the bot
        if message.author == self.user:
            return
        
        else: 
            Username = str(message.author.name)
            #storing each user's message record into corresponding text file
            txt_file_path = Username +'.txt'

            #Generate Big5 traits result
            if message.content.startswith('result'):
                await message.channel.send('Generating results...')
                # Create an instance of TraitsResult
                traits_result = TraitsResult(txt_file_path, tokenizer, model)
                # Call the predict method on the instance
                list_of_results = traits_result.predict()
                result = '\n'.join(list_of_results)
                await message.channel.send(f"{Username}'s BIG5 personality score is...")
                await message.channel.send(result)

                #create a dictionary to facilitate writing data into the csv file
                result_dictionary = {}
                result_dictionary['username'] = Username
                for item in list_of_results:
                    key, value = item.split(': ')
                    result_dictionary[key] = float(value)

                print(result_dictionary)

                with open('Userdata.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    row = [result_dictionary.get(header, '') for header in headers]
                    writer.writerow(row)
                    #there will be repeated data

            #friend recommedations
            if message.content.startswith('Find similar individual'):
                data = pd.read_csv("Userdata.csv")
                # Get user's score
                user_traits = data[data['username'] == Username].iloc[0, 1:].values

                # Calculate Euclidean distances from user's traits to all other users' traits excluding him/herself
                data['distance'] = data.apply(lambda row: np.linalg.norm(row[1:] - user_traits) if row['username'] != Username else np.inf, axis=1)

                # Find the user with the closest personality to the user excluding him/herself
                closest_person = data.loc[data['distance'].idxmin()]['username']

                await message.channel.send(f"The individual with the closest personality to {Username} is: {closest_person}")
                await message.channel.send('Feel free to dm them and start a conversation!')

            if message.content.startswith('Find contrasting individual'):
                data = pd.read_csv("Userdata.csv")
                # Get user's score
                user_traits = data[data['username'] == Username].iloc[0, 1:].values

                # Calculate Euclidean distances from user's traits to all other users' traits excluding him/herself
                data['distance'] = data.apply(lambda row: np.linalg.norm(row[1:] - user_traits), axis=1)

                # Find the user with the most contrasting personality to the user excluding him/herself
                distinct_person = data.loc[data['distance'].idxmax()]['username']

                await message.channel.send(f"The individual with the most contrasting personality to {Username} is: {distinct_person}")
                await message.channel.send('Feel free to dm them and start a conversation!')

            #Allow users to delete chat records
            elif message.content.startswith('delete'):
                await message.channel.send('Deleting...')
                # Open the file in write mode to truncate its content
                with open(txt_file_path, "w") as file:
                    # This will truncate the file, effectively deleting its contents
                    pass
                await message.channel.send('Deletion Complete')

            else: 
                with open(txt_file_path, "a") as file:
                    file.write(message.content + ' ')



#set default gateway intents
intents = discord.Intents.default()
intents.message_content = True
client = Client(command_prefix='!',intents=intents)


client.run(token)