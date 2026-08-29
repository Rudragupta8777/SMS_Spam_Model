"""
Template generator for code-mixed Indian SMS, in English / Hindi (Devanagari) / Hinglish.

WHY THIS IS NEEDED
------------------
The whole premise of this project is catching regional code-mixed spam, but the only real
code-mixed spam available is indian_scam.csv: 17 Hinglish + 21 Hindi messages. You cannot train
a classifier on 38 examples. This module fills that gap with templates grounded in the smishing
categories that actually circulate in India (KYC freeze, customs parcel, lottery, UPI/OTP theft,
loan approval, electricity disconnection, Aadhaar/SIM, fake job offers).

TWO HONEST CAVEATS
------------------
1. Synthetic data is *train-only*. Scoring a model on generated text measures how well it
   learned the generator, not the real world. evaluate.py therefore keeps real messages
   (indian_scam.csv) as a separate held-out slice and reports that number separately.
2. Templates cannot capture everything real spam does. They widen coverage of known patterns;
   the telemetry loop (fetch_telemetry.py) is what brings in genuinely novel ones.

HARD NEGATIVES
--------------
Half the value here is `legit` output, not spam. A real bank OTP, a real delivery update and a
real payment receipt all contain the same surface features as a scam - rupee amounts, six-digit
codes, order ids, a sense of time pressure. A model trained on spam-with-numbers versus
chit-chat-without-numbers learns "contains a number => spam" and flags every real OTP. These
templates deliberately share that surface with the spam ones, so the model is forced to key on
what actually differs: who is asking, for what action, and whether an unsolicited link or
payment is required to receive something you never initiated.
"""
import random

BRANDS = ["SBI", "HDFC", "ICICI", "Axis Bank", "Kotak", "PNB", "Paytm", "PhonePe", "GPay",
          "Amazon", "Flipkart", "Myntra", "Meesho", "BlueDart", "Delhivery", "IndiaPost"]
SHORT_LINKS = ["bit.ly/{c}", "tinyurl.com/{c}", "rb.gy/{c}", "cutt.ly/{c}", "t.ly/{c}",
               "kyc-verify{c}.in", "secure-{c}.xyz", "reward{c}.online", "claim-{c}.top"]
OFFICIAL = ["sbi.co.in", "hdfcbank.com", "icicibank.com", "amazon.in", "flipkart.com",
            "bluedart.com", "indiapost.gov.in", "paytm.com"]


def _code(rng, n=6):
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(n))


def _amount(rng):
    return rng.choice(["499", "899", "1,250", "2,499", "4,999", "9,999", "15,000",
                       "25,000", "50,000", "1,00,000", "5,00,000", "10,00,000"])


def _otp(rng):
    return str(rng.randint(100000, 999999))


# Deliberate obfuscations real spam uses to dodge keyword filters. The char n-grams in
# text_features.py are what let the model still recognise these.
def _obfuscate(text, rng):
    if rng.random() < 0.30:
        subs = {"o": "0", "i": "1", "e": "3", "a": "@", "s": "$"}
        ch = rng.choice(list(subs))
        text = text.replace(ch, subs[ch], rng.randint(1, 2))
    if rng.random() < 0.20:
        text = text.upper() if rng.random() < 0.4 else text
    if rng.random() < 0.25:
        text = text.replace(" ", "  ", 1)
    return text


SPAM_TEMPLATES = [
    # --- KYC / account freeze ---
    "Dear customer, your {brand} account will be blocked today. Complete KYC now: {link}",
    "आपका {brand} खाता आज बंद हो जाएगा। तुरंत KYC पूरा करें: {link}",
    "Aapka {brand} account band ho jayega. Turant KYC update kare: {link}",
    "{brand} KYC pending hai! Account suspend hone se bachane ke liye abhi click kare {link}",
    "प्रिय ग्राहक, आपका KYC अपूर्ण है। खाता बंद होने से बचाएं: {link}",

    # --- Parcel / customs ---
    "Your {brand} parcel is held at customs. Pay Rs {amt} to release: {link}",
    "Aapka parcel customs me atka hai. Rs {amt} pay karke release kare: {link}",
    "आपका पार्सल सीमा शुल्क में रुका है। {amt} रुपये का भुगतान करें: {link}",
    "{brand} delivery failed. Reschedule by paying Rs {amt} here {link}",
    "Parcel delivery pending hai, address confirm kare aur Rs {amt} de: {link}",

    # --- Lottery / prize ---
    "Congratulations! You won Rs {amt} in {brand} lucky draw. Claim now {link}",
    "बधाई हो! आपने {amt} रुपये जीते हैं। अभी दावा करें: {link}",
    "Badhai ho! Aapne {amt} rupaye ki lottery jeeti hai. Abhi claim kare {link}",
    "Bhai lottery jeet gaya! Rs {amt} ka inaam turant claim karo {link}",
    "WINNER! {brand} ne aapko Rs {amt} diya hai. Paise lene ke liye click {link}",

    # --- UPI / OTP theft ---
    "Your account is debited Rs {amt}. If not you, call {phone} immediately",
    "आपके खाते से {amt} रुपये कट गए। यदि आपने नहीं किया तो कॉल करें {phone}",
    "Aapke account se Rs {amt} cut gaye. Agar aapne nahi kiya to turant call kare {phone}",
    "UPI block hone wala hai. OTP {otp} share kare warna account band {link}",
    "Share OTP {otp} to stop unauthorized transaction of Rs {amt} now",

    # --- Loan ---
    "Instant loan Rs {amt} approved! No documents needed. Apply {link}",
    "तुरंत {amt} रुपये का लोन मंजूर! कोई कागज नहीं। आवेदन करें {link}",
    "Turant Rs {amt} ka loan approved hai, bina document. Abhi apply kare {link}",
    "Personal loan {amt} bina CIBIL check. 2 minute me paisa {link}",

    # --- Electricity / utility disconnection ---
    "Dear user your electricity will be disconnected tonight. Pay now {link}",
    "आपकी बिजली आज रात काट दी जाएगी। तुरंत भुगतान करें {link}",
    "Aaj raat aapki bijli kat jayegi, pichla bill pending. Abhi pay kare {link}",
    "Electricity connection band hoga 9 baje. Update kare {link}",

    # --- Aadhaar / SIM / PAN ---
    "Your SIM will be deactivated in 24 hours. Link Aadhaar here {link}",
    "आपका सिम 24 घंटे में बंद हो जाएगा। आधार लिंक करें {link}",
    "Aapka SIM 24 ghante me band. Aadhaar link kare turant {link}",
    "PAN card {brand} account se link nahi hai. Abhi kare warna block {link}",

    # --- Job / work from home ---
    "Work from home! Earn Rs {amt} daily. Join now {link}",
    "घर बैठे {amt} रुपये रोज कमाएं। अभी जुड़ें {link}",
    "Ghar baithe Rs {amt} roz kamaye, koi experience nahi chahiye {link}",
    "{brand} me job nikli hai, salary {amt}. Registration fee de kar apply {link}",

    # --- Rewards / cashback ---
    "You have {amt} reward points expiring today! Redeem {link}",
    "आपके {amt} रिवॉर्ड पॉइंट आज समाप्त हो रहे हैं। भुनाएं {link}",
    "Aapke {amt} reward point aaj expire ho rahe hai. Turant redeem kare {link}",
    "{brand} cashback Rs {amt} pending hai. Claim karne ke liye click {link}",

    # --- Investment ---
    "Double your money in 30 days! Guaranteed returns. Invest {link}",
    "30 din me paisa double! Guaranteed profit. Abhi invest kare {link}",
    "Stock tip: buy today, {amt} profit guaranteed. Join our group {link}",

    # --- Tamil script + Tanglish ---
    # The multilingual corpus has no Tamil column at all, so without these the model has
    # literally zero Tamil training signal and misses Tamil smishing outright (verified
    # on-device before these were added). The brief names Tanglish explicitly.
    "உங்கள் வங்கிக் கணக்கு இன்று முடக்கப்படும். உடனே KYC முடிக்கவும்: {link}",
    "உங்கள் பார்சல் சுங்கத்தில் நிறுத்தப்பட்டுள்ளது. ரூ {amt} செலுத்தவும்: {link}",
    "வாழ்த்துக்கள்! நீங்கள் ரூ {amt} வென்றுள்ளீர்கள். இப்போதே பெறுங்கள்: {link}",
    "உங்கள் சிம் 24 மணி நேரத்தில் முடக்கப்படும். ஆதார் இணைக்கவும்: {link}",
    "உங்கள் மின்சாரம் இன்று இரவு துண்டிக்கப்படும். இப்போதே செலுத்துங்கள்: {link}",
    "உடனடி கடன் ரூ {amt} அங்கீகரிக்கப்பட்டது! ஆவணங்கள் தேவையில்லை: {link}",
    "வீட்டில் இருந்து வேலை! தினமும் ரூ {amt} சம்பாதிக்கவும்: {link}",
    "Ungal bank account inru mudakkappadum, udane KYC mudikkavum {link}",
    "Ungal parcel customs la nikkuthu, Rs {amt} pay pannunga {link}",
    "Vaazhthukkal! Neenga Rs {amt} vennutteenga, ipove claim pannunga {link}",
    "Ungal SIM 24 mani nerathil block agum, Aadhaar link pannunga {link}",
    "Ungal current inniki night cut aagum, ipove bill kattunga {link}",
    "Udane loan Rs {amt} approve aagiduchu, document venam {link}",
    "Veetla irunthu velai! Dinam Rs {amt} sambadhikkalam {link}",
]

# Legit messages sharing the same surface features - the hard negatives that protect precision.
LEGIT_TEMPLATES = [
    "Your {brand} OTP is {otp}. Valid for 10 minutes. Do not share with anyone.",
    "आपका {brand} ओटीपी {otp} है। किसी के साथ साझा न करें।",
    "Aapka {brand} OTP {otp} hai. Kisi ke saath share na kare.",
    "Rs {amt} debited from your {brand} account ending {last4} on {date}. Bal: Rs {amt2}",
    "आपके {brand} खाते से {amt} रुपये निकाले गए। शेष राशि {amt2} रुपये।",
    "Aapke {brand} account se Rs {amt} debit hua. Balance Rs {amt2}.",
    "Rs {amt} credited to your {brand} account ending {last4}. Available balance Rs {amt2}",
    "Your {brand} order #{order} has been shipped and arrives by {date}.",
    "आपका {brand} ऑर्डर #{order} भेज दिया गया है, {date} तक पहुंचेगा।",
    "Aapka {brand} order #{order} ship ho gaya hai, {date} tak pahunchega.",
    "Your {brand} order #{order} is out for delivery today. Track at {official}",
    "{brand} order #{order} delivered successfully. Thank you for shopping with us.",
    "Payment of Rs {amt} received against invoice #{order}. Thank you.",
    "{amt} रुपये का भुगतान प्राप्त हुआ। धन्यवाद।",
    "Rs {amt} ka payment mil gaya hai. Dhanyavaad.",
    "Your electricity bill of Rs {amt} is due on {date}. Pay via {official}",
    "आपका बिजली बिल {amt} रुपये {date} को देय है।",
    "Aapka bijli bill Rs {amt} {date} ko due hai. {official} par pay kare.",
    "Your {brand} statement for this month is ready. View at {official}",
    "Reminder: your appointment is scheduled for {date} at {time}.",
    "आपकी अपॉइंटमेंट {date} को {time} बजे है।",
    "Aapki appointment {date} ko {time} baje hai.",
    "{brand} recharge of Rs {amt} successful. Validity till {date}.",
    "Aapka {brand} recharge Rs {amt} safal raha. Validity {date} tak.",
    "Your refund of Rs {amt} for order #{order} has been processed to source account.",
    "आपका {amt} रुपये का रिफंड संसाधित कर दिया गया है।",
    "Aapka Rs {amt} ka refund process ho gaya hai, 3-5 din me aayega.",
    "Low balance alert: your {brand} account balance is Rs {amt2}.",
    "Your monthly {brand} EMI of Rs {amt} is due on {date}.",
    "Aapki {brand} EMI Rs {amt} {date} ko due hai. Time par pay kare.",

    # Tamil / Tanglish legit. These exist for the same reason the Hindi ones do: adding Tamil
    # SPAM without Tamil HAM would teach the model "Tamil script => spam" instead of teaching it
    # what Tamil spam actually looks like.
    "உங்கள் {brand} OTP {otp}. யாருடனும் பகிர வேண்டாம்.",
    "உங்கள் ஆர்டர் #{order} அனுப்பப்பட்டது, {date} அன்று வந்துவிடும்.",
    "ரூ {amt} உங்கள் {brand} கணக்கிலிருந்து எடுக்கப்பட்டது. இருப்பு ரூ {amt2}.",
    "ரூ {amt} உங்கள் {brand} கணக்கில் வரவு வைக்கப்பட்டது.",
    "உங்கள் மின்கட்டணம் ரூ {amt} {date} அன்று செலுத்த வேண்டும்.",
    "உங்கள் ஆர்டர் #{order} வெற்றிகரமாக வழங்கப்பட்டது. நன்றி.",
    "Ungal {brand} OTP {otp}. Yaarukkum share panna vendam.",
    "Ungal order #{order} ship aagiduchu, {date} varum.",
    "Rs {amt} ungal {brand} account la irunthu debit aachu. Balance Rs {amt2}.",
    "Ungal current bill Rs {amt} {date} ku kattanum.",
    "Ungal refund Rs {amt} process aagiduchu, 3-5 naal la varum.",
]


def _fill(tpl, rng):
    return tpl.format(
        brand=rng.choice(BRANDS),
        link=rng.choice(SHORT_LINKS).format(c=_code(rng)),
        official=rng.choice(OFFICIAL),
        amt=_amount(rng),
        amt2=_amount(rng),
        otp=_otp(rng),
        phone="{}{}".format(rng.choice([6, 7, 8, 9]), rng.randint(100000000, 999999999)),
        last4=str(rng.randint(1000, 9999)),
        order=str(rng.randint(10000000, 99999999)),
        date=rng.choice(["12 Jan", "03 Feb", "21 Mar", "08 Apr", "17 May", "29 Jun",
                         "tomorrow", "today", "kal", "aaj"]),
        time=rng.choice(["10:30", "11:00", "14:15", "16:45", "18:00"]),
    )


def generate(n_spam=4000, n_ham=3000, seed=42):
    """Returns (spam_texts, ham_texts). Seeded so the corpus is reproducible."""
    rng = random.Random(seed)
    spam, ham = set(), set()

    guard = 0
    while len(spam) < n_spam and guard < n_spam * 60:
        t = _obfuscate(_fill(rng.choice(SPAM_TEMPLATES), rng), rng)
        spam.add(t)
        guard += 1

    guard = 0
    while len(ham) < n_ham and guard < n_ham * 60:
        ham.add(_fill(rng.choice(LEGIT_TEMPLATES), rng))
        guard += 1

    return sorted(spam), sorted(ham)


if __name__ == "__main__":
    s, h = generate()
    print(f"generated {len(s)} spam, {len(h)} legit")
    for t in s[:5]:
        print("  SPAM :", t)
    for t in h[:5]:
        print("  LEGIT:", t)
