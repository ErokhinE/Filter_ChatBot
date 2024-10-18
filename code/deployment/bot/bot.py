import telebot
import time
import requests





# creating bot instance
bot = telebot.TeleBot('7651842902:AAGSFRba28s8KkXmhxtwT_48eLnSp1A0Pf8')

# /start command
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Hello there, I am a bot to help your chat to stay positive and clean\nCheck \help command to see what I can")

# /help command
@bot.message_handler(commands=['help'])
def help(message):
    bot.reply_to(message, "I check each message in this chat for negativity and if ir is labeled negative\nI have rights to mute the sender based on the previous behaviour.\n" +
                 "Also for administrator I have such commands as \mute and \unmute if my capabilities are not enough to detect unethical messages.")
# /kick command
@bot.message_handler(commands=['kick'])
def kick_user(message):
    if message.reply_to_message:
        chat_id = message.chat.id
        user_id = message.reply_to_message.from_user.id
        user_status = bot.get_chat_member(chat_id, user_id).status
        if user_status == 'administrator' or user_status == 'creator':
            bot.reply_to(message, "Cannot kick an admin")
        else:
            bot.kick_chat_member(chat_id, user_id)
            bot.reply_to(message, f"User {message.reply_to_message.from_user.username} was kicked.")
    else:
        bot.reply_to(message, "This command is designed to kick the user for whose message you replied.")

# /mute command
@bot.message_handler(commands=['mute'])
def mute_user(message):
    if message.reply_to_message:
        chat_id = message.chat.id
        user_id = message.reply_to_message.from_user.id
        user_status = bot.get_chat_member(chat_id, user_id).status
        if user_status == 'administrator' or user_status == 'creator':
            bot.reply_to(message, "Cannot mute admin")
        else:
            duration = 1 # Значение по умолчанию - 1 минута
            args = message.text.split()[1:]
            if args:
                try:
                    duration = int(args[0])
                except ValueError:
                    bot.reply_to(message, "Incorrect time format.")
                    return
                if duration < 1:
                    bot.reply_to(message, "Time should be a positive number.")
                    return
                if duration > 1440:
                    bot.reply_to(message, "Maximum duration is 1 day.")
                    return
            bot.restrict_chat_member(chat_id, user_id, until_date=time.time()+duration*60)
            bot.reply_to(message, f"User {message.reply_to_message.from_user.username} was muted for {duration} minutes.")
    else:
        bot.reply_to(message, "This command is designed to mute the user for whose message you replied.")

# /unmute command
@bot.message_handler(commands=['unmute'])
def unmute_user(message):
    if message.reply_to_message:
        chat_id = message.chat.id
        user_id = message.reply_to_message.from_user.id
        bot.restrict_chat_member(chat_id, user_id, can_send_messages=True, can_send_media_messages=True, can_send_other_messages=True, can_add_web_page_previews=True)
        bot.reply_to(message, f"User {message.reply_to_message.from_user.username} was unmuted.")
    else:
        bot.reply_to(message, "This command is designed to unmute the user for whose message you replied.")

def check_message(message):
    response = requests.get('', data=message.text)
    if response.status_code == 200:
        result = response.json()
        if result['label'] == 'NEGATIVE':
            return True
        return False
    else:
        raise requests.exceptions.ConnectionError('Something went wrong')
    
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    if check_message(message):
        chat_id = message.chat.id
        user_id = message.from_user.id
        user_status = bot.get_chat_member(chat_id, user_id).status
        if user_status == 'administrator' or user_status == 'creator':
            bot.reply_to(message, "Cannot mute an admin.")
        else:
            duration = 1
            bot.reply_to(message, f"Пользователь {message.from_user.username} замучен на {duration} минут.")



# Запускаем бота
bot.infinity_polling(none_stop=True)
