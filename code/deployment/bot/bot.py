import telebot
import time
import requests


# creating bot instance
bot = telebot.TeleBot('paste_token')

# /start command
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Hello there, I am a bot to help your chat to stay positive and clean\nCheck /help command to see what I can")

# /help command
@bot.message_handler(commands=['help'])
def help(message):
    bot.reply_to(message, "I check each message in this chat for negativity and if ir is labeled negative\nI have rights to mute the sender based on the previous behaviour.\n" +
                 "Also for administrator I have such commands as /report_spam and /report_negativity if my capabilities are not enough to detect unethical messages.")

# /report commands
@bot.message_handler(commands=['report_spam', 'report_negativity'])
def report(message):
    if message.reply_to_message:
        chat_id = message.chat.id
        user_id = message.reply_to_message.from_user.id
        user_status = bot.get_chat_member(chat_id, user_id).status
        if user_status == 'administrator' or user_status == 'creator':
            bot.reply_to(message, "Cannot mute admin")
        else:
            duration = 1 # Значение по умолчанию - 1 минута
            # bot.restrict_chat_member(chat_id, user_id, until_date=time.time()+duration*60)
            if message.text.split()[0] == '/report_spam':
                bot.reply_to(message, f"User {message.reply_to_message.from_user.username} was muted for {duration} minutes. For spam")
            else:
                bot.reply_to(message, f"User {message.reply_to_message.from_user.username} was muted for {duration} minutes. For negative message")
    else:
        bot.reply_to(message, "This command is designed to mute the user for whose message you replied.")

# /unmute command
@bot.message_handler(commands=['unmute'])
def unmute_user(message):
    if message.reply_to_message:
        chat_id = message.chat.id
        user_id = message.reply_to_message.from_user.id
        # bot.restrict_chat_member(chat_id, user_id, can_send_messages=True, can_send_media_messages=True, can_send_other_messages=True, can_add_web_page_previews=True)
        bot.reply_to(message, f"User {message.reply_to_message.from_user.username} was unmuted.")
    else:
        bot.reply_to(message, "This command is designed to unmute the user for whose message you replied.")

def check_message(message):
    if message.content_type == 'text':
        data = {'text': message.text}
        response = requests.post('http://flask:5000/predict_label', json=data)
    else:
        print('got audio')
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        response = requests.post('http://flask:5000/predict_voice', files={"the_file": downloaded_file})
    if response.status_code == 200:
        result = response.json()
        if result['label'] == 'NEGATIVE' or result['label'] == 'spam':
            return result['label']
        return False
    else:
        print(response.text)
        raise requests.exceptions.ConnectionError('Something went wrong')
    
@bot.message_handler(func=lambda message: True, content_types=['voice', 'text'])
def handle_message(message):
    recognized = check_message(message)
    if recognized:
        chat_id = message.chat.id
        user_id = message.from_user.id
        user_status = bot.get_chat_member(chat_id, user_id).status
        if user_status == 'administrator' or user_status == 'creator':
            bot.reply_to(message, "Cannot mute an admin.")
        else:
            duration = 1
            if recognized == 'spam':
                bot.reply_to(message, f"User {message.from_user.username} was muted for {duration} minutes. For spam")
            else:
                bot.reply_to(message, f"User {message.from_user.username} was muted for {duration} minutes. For negative message")
            

# starting the bot
bot.infinity_polling(none_stop=True)
