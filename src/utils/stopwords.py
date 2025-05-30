from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Expand list of stopwords to remove industry-specific noise
# and months of the year
iata_stopwords = [
    "sky", "flight", "air", "aviation", "iata", "sector", "industry",
    "annual", "review", "general", "director", "transport", "association",
    "chief", "executive", "officer", "bisignani", "giovanni", "ceo", "chair"
    "board", "governor", "governors", "chairman", "president", "mr", "ms", "miss", "mrs", "dr",
    "airlines", "airline", "business", "businesses", "airways", "airway", 
    "representing", "representatives", "representative", "fax", "tel", "group", 
    "offices", "geneva", 
]
iata_member_airlines = [
    "abx", "aegean", "aer", "aero", "aeroflot", "aeroitalia", "aerolineas", "aeromexico",
    "africa", "afrijet", "air", "airlines", "algerie", "albastar", "alaska", "allied",
    "alma", "almasria", "alme", "almo", "ameli", "american", "ana", "angkor", "anguilla",
    "apg", "arabia", "arabian", "argentina", "arkia", "asia", "asiana", "asky", "asl", "atlanta", "atlantic",
    "atlas", "austrian", "aviacion", "avianca", "avion", "azerbaijan", "azores", "azul",
    "badr", "bahamasair", "bamboo", "bangkok", "batik", "belarusian", "belavia", "biman",
    "binter", "boa", "boliviana", "botswana", "braathens", "britannia", "british", "brussels",
    "bulgaria", "cairo", "camair", "cambodia", "canada", "canarias", "capital", "cargo",
    "cargojet", "cargolux", "caribbean", "carpatair", "cathay", "cayman", "cebu", "cemair",
    "chalair", "challenge", "changan", "china", "cityjet", "clic", "compagnie", "condor",
    "congo", "copa", "corendon", "corsair", "costa", "croatia", "cubana", "cyprus", "czech",
    "dat", "delta", "dh", "dhl", "discover", "dreamjet", "east", "eastern", "edelweiss", "egyptair",
    "el", "emirates", "ethiopian", "etihad", "euroatlantic", "european", "eurowings", "eva",
    "eznis", "express", "fedex", "fiji", "finnair", "fly", "flybaghdad", "flydubai", "flyegypt",
    "flynas", "flynamibia", "flyone", "france", "freebird", "french", "fuzhou", "garuda", "georgian",
    "german", "getjet", "globalx", "gol", "greater", "group", "gulf", "gx", "hahnair", "hainan",
    "hawaiian", "hebei", "hello", "heston", "hi", "himalaya", "hong", "iberia", "iberojet",
    "ibom", "icelandair", "ikar", "indigo", "international", "iran", "irani", "iranian",
    "iranaseman", "island", "islander", "israel", "israir", "ita", "jair", "jal", "japan",
    "jazeera", "jeju", "jetblue", "jin", "jordan", "juneyao", "kam", "kenya", "klm", "km",
    "korean", "kunming", "kuwait", "la", "lam", "lan", "lao", "latam", "link", "lion", "loong", "lot",
    "lucky", "lufthansa", "luxair", "madagascar", "malaysia", "malta", "mandarin", "martinair",
    "masair", "mauritania", "mea", "miat", "mng", "montenegro", "myanmar", "namibia", "national",
    "neos", "nesma", "new", "nile", "nippon", "nok", "nordstar", "nordwind", "nouvelair", "okay",
    "olympic", "oman", "overland", "pacific", "pakistan", "pal", "paranair", "pegasus", "peru",
    "philippine", "plus", "polar", "populair", "poste", "precision", "privilege", "pt", "qantas",
    "qatar", "qazaq", "red", "republic", "rossiya", "royal", "russian", "rwandair", "s7", "safair",
    "salam", "sas", "sata", "saudi", "scat", "scoot", "sea", "sf", "shandong", "shanghai",
    "shenzhen", "sichuan", "silk", "singapore", "sky", "smartavia", "smartwings", "solomon",
    "somon", "south", "spicejet", "srilankan", "starlux", "style", "suparna", "sun", "suncountry",
    "sunexpress", "swiftair", "swiss", "syrianair", "taag", "taca", "tahiti", "tag", "tap", "tarom", "tassili",
    "thai", "tianjin", "tibet", "tui", "tunisair", "turkish", "tus", "tway", "ukraine", "uls",
    "uni", "united", "ural", "urumqi", "us", "uzbekistan", "vanuatu", "vietjet", "vietnam",
    "virgin", "vistara", "voepass", "volaris", "volotea", "vueling", "wamos", "west", "westair",
    "westjet", "white", "wideroe", "world", "xiamen", "yto", "linhas", "austral","kong", "ecuador", 
    "atm", "aéreas", "members", "blue", "members",
    "3", "4", "51", "adria", "aegean", "aer", "aereas", "aero", "aeroflot",
    "aerolineas", "aeromexico", "africa", "african", "aigle", "air",
    "airbridgecargo", "aircalin", "airline", "airlines", "airlink", "airtour",
    "airways", "al", "alaska", "algerie", "alitalia", "allied", "almasria",
    "american", "ana", "angkor", "arabia", "arabian", "argentina", "argentinas",
    "arik", "arkia", "aseman", "asiana", "associate", "astana", "atlantic",
    "atlas", "atlasglobal", "austral", "australia", "austrian", "aviacion",
    "avianca", "aviation", "azerbaijan", "azul", "azur", "açores", "aérea",
    "aéreas", "aéreos", "bahamasair", "baltic", "bangkok", "bangladesh", "batik",
    "belarusian", "belavia", "bh", "biman", "binter", "blue", "bmi", "boa",
    "boliviana", "botswana", "braathens", "brasil", "brazilian", "british",
    "brunei", "brussels", "bulgaria", "burkina", "cabo", "cairo", "cal",
    "caledonie", "camairco", "cambodia", "canada", "canarias", "capital",
    "caraibes", "cargo", "cargojet", "cargolux", "caribbean", "carpatair",
    "cathay", "cayman", "cemair", "china", "cityjet", "cityline", "cobalt",
    "colombia", "comair", "condor", "copa", "corendon", "corsair", "corsica",
    "costa", "croatia", "cubana", "czech", "de", "del", "delta", "dhl", "dragon",
    "east", "eastar", "eastern", "ecuador", "egyptair", "el", "emirates",
    "ethiopian", "etihad", "euroatlantic", "europa", "european", "eurowings",
    "eva", "evelop", "express", "fedex", "fiji", "finnair", "fly", "flybe",
    "flydubai", "flyegypt", "france", "freebird", "garuda", "georgian", "germania",
    "gol", "group", "gulf", "gx", "hahn", "hainan", "hawaiian", "hebei", "hi",
    "hong", "iberia", "icelandair", "india", "indonesia", "inselair", "interjet",
    "international", "iran", "israeli", "israir", "italy", "japan", "jat",
    "jazeera", "jeju", "jet", "jetblue", "jordan", "jordanian", "jsc", "juneyao",
    "kenya", "kingfisher", "kish", "klm", "kong", "korean", "koryo", "kuwait",
    "lacsa", "lam", "lamlinhas", "lan", "lao", "latam", "lauda", "libyan", "lines",
    "lingus", "linhas", "list", "lite", "lot"
]
iata_board = [
    "willie", "walsh", "pieter", "elbers", "michael", "rousseau", "benjamin", "smith",
    "campbell", "wilson", "robert", "isom", "shinichi", "inoue", "patrick", "healy",
    "pedro", "heilbron", "jasmin", "bajić", "mesfin", "tasew", "bekele", "richard",
    "zhu", "tao", "luis", "gallego", "martín", "mitsuko", "tottori", "marjan", "rintel",
    "walter", "cho", "roberto", "alvo", "carsten", "spohr", "izham", "ismail",
    "munkhtamir", "batbayar", "mohamad", "el-hout", "mehmet", "tevfik", "nane",
    "badr", "mohammed", "al-meer", "abdelhamid", "addou", "yvonne", "manzi", "makolo",
    "anco", "van", "der", "werff", "ibrahim", "al-omar", "ahmet", "bolat", "naresh",
    "goyal", "gerald", "grinstein", "sigurdur", "helgason", "liu", "shaoyong", "isao",
    "kaneko", "jorgen", "lindegaard", "carlos", "luiz", "martins", "wolfgang", "mayrhuber",
    "robert", "milton", "atef", "abdel", "hamid", "mostafa", "titus", "naikuni", "valery",
    "okulov", "nelson", "ramiz", "vagn", "soerensen", "jean-cyril", "spinetta", "gerard",
    "arpey", "khaled", "ben-bakr", "david", "bronczek", "philip", "chen", "chew", "choon",
    "seng", "yang", "ho", "cho", "fernando", "conte", "enrique", "cueto", "geoff", "dixon",
    "samer", "majali", "fernando", "pinto", "toshiyuki", "shinmachi", "alan", "joyce",
    "akbar", "baker", "khalid", "abdullah", "almolhem", "richard", "anderson", "tawfik",
    "assy", "andrés", "conesa", "peter", "davies", "german", "efromovich", "christoph",
    "franz", "sameh", "ahmed", "zaky", "hefny", "tewolde", "gebremariam", "goh", "choon",
    "phong", "rickard", "gustafson", "peter", "hartman", "james", "hogan", "harry",
    "hohmeister", "temel", "kotil", "phạm", "ngọc", "minh", "mbuvi", "ngunze", "masaru",
    "onishi", "douglas", "parker", "vitaly", "saveliev", "si", "xian", "min", "jeffery",
    "smisek", "robin", "hayes", "sebastian", "mikosz", "saleh", "nasser", "al-jasser",
    "yuji", "akasaka", "franck", "terner", "tan", "wangeng", "alexandre", "juniac",
    "tony", "tyler", "andres", "conesa", "maria", "jose", "hidalgo", "gutierrez",
    "christine", "ourmières-widener", "badr", "al-meer"
]

months = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]
month_abbr = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]
CUSTOM_STOPWORDS = sorted(list(ENGLISH_STOP_WORDS.union(iata_stopwords, iata_member_airlines, iata_board, months, month_abbr)))
