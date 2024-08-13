"""
Reference key word patterns for parking detection:
Pattern descriptions:
core= regex + url + www.url
attributes: regex, identifier, expcluded core keywords (indexes of excluded), excluded words
"""

import os
from os.path import join
import re
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
from itertools import groupby
from utils import load_obj
import jieba
from config import PLOG

TARGET_LANGUAGES = ["en", "ru", "zh-cn", "zh-tw", "es", "fr", "sv", "de", "et", "it", "da", "sk"]
# language codes definition:  https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes

DICO_PATTERN_ENG = {"core": ["( |^)domain", "( |^)page(?!s)", "site(?!s)"],
                    "attributes": [["parked", "p1"],
                                   ["owne", "p2", [1, 2]],  # "owner, owned"
                                   ["sale", "p3", [1, 2]],
                                   ["(?<!right)(?<!rights) reserved", "p4"],
                                   ["registered", "p5", [1, 2]],
                                   ["generated", "p6", [0, 2, 3, 4]],
                                   ["registra", "p7"],
                                   ["unauthorized", "p8"],
                                   ["expired", "p9"],
                                   ["available soon", "p10"],
                                   ["unavailable", "p11"],
                                   ["not available", "p11"],
                                   ["available", "p12", [1, 2, 3, 4]],
                                   ["create", "p13", [1, 2, 3, 4]],
                                   ["coming soon", "p14"],
                                   ["parking", "p15"],
                                   ["opening soon", "p16"],
                                   ["maintenance", "p17"],
                                   ["maintaining", "p18"],
                                   ["under development", "p19"],
                                   ["buy", "p20", [1, 2]],
                                   ["purchas", "p21", [1, 2]],
                                   ["suspended", "p22"],
                                   ["undergoing", "p23"],
                                   ["censored", "p24"],
                                   # ["hosting", "parked"],
                                   ["blocked", "p25"],
                                   ["deleted", "p26"],
                                   ["acquire", "p27"],
                                   ["establish", "p28"],
                                   ["display page", "p29"],
                                   ["taken", "p30"],
                                   ["not linked", "p31"],
                                   ["upload", "p32"],
                                   ["propagated", "p33"],
                                   ["default", "p34"],
                                   ["no website here", "p35", [0, 1, 3, 4]],
                                   ["cannot be displayed", "p36", [0, 2, 3, 4]],
                                   ["construction", "p37"],
                                   ]}

DICO_PATTERN_FR = {"core": ["domaine(?!s)", "( |^)page(?!s)", "( |^)site(?!s)"],
                   "attributes": [["parking", "parked"],
                                  ["(?<!droit)(?<!droits) reservé", "parked"],
                                  ["prochainement", "construction"],
                                  ["développement", "construction"],
                                  ["maintenance", "construction"],
                                  ["redirige pas", "construction"],
                                  ["moderation""construction"],
                                  ["obsolète", "expired"],
                                  ["expired", "expired"],
                                  ["supprimé", "parked"],
                                  ["enregistr", "parked", [1, 2]],
                                  ["aquérir", "sale"],
                                  ["suspendu", "parked"],
                                  ["configurez", "parked"],
                                  ["indisponible", "parked"],
                                  ["pas disponible", "parked"],
                                  ["vente", "sale"],
                                  ["vendre", "sale"],
                                  ["propriétaire", "owner", [1, 2]],
                                  # ["héberge", "parked"],  # heberger, hebergement  but not hebergé
                                  ["parked", "parked"],
                                  ["registrar", "parked"],
                                  ["cré[ée]", "parked"],
                                  ["construction", "construction"],
                                  ["domaine", "p1", [0, 1, 2]],
                                  ["déposé", "parked"]
                                  ]}
DICO_PATTERN_ES = {"core": ["dominio", "p[áa]gina(?!s)", "( |^)sitio(?!s)"],
                   "attributes": [["pedido", "order"],  # parked
                                  ["venta", "sale", [1, 2]],  # sale
                                  ["estacionado", "parked"],  # parked
                                  ["mantenimiento", "parked"],  # maintenance
                                  ["domaine", "parked"],  # domain
                                  ["disponible", "parked"],  # domain
                                  ["(?<!derecho)(?<!derechos) reserva", "parked"],  # reserved
                                  ["compra", "sale", [1, 2]],  # purchase
                                  ["venta", "sale"],  # sale
                                  ["suspendido", "parked"],  # suspended
                                  ["proprietario", "parked", [1, 2]],  # owner   # not propias (cookies)
                                  ["hosting", "parked"],  # hosting
                                  ["alojada", "parked"],  # hosted
                                  ["registr", "parked", [1, 2, 3, 4]],  # registered, registrys, registrado
                                  ["parking", "parked"],  # parked
                                  ["activ[oa]", "parked"],  # inactive
                                  ["propagado", "parked"],  # propagated
                                  ["defecto", "parked"]  # default
                                  ]}
DICO_PATTERN_CN = {"core": ["域名", "(?<!账号网)站", "页", "頁"],  # add 主机 (host?)  #(?<!账号网)
                   "attributes": [
                       ["购", "sale", [1, 2, 3]],  # purchase
                       ["^购买网站$", "sale"],  # purchase website
                       ["在出售", "parked"],  # on sale
                       ["无法访问", "parked"],  # innaccessible
                       ["无法进行访问", "parked"],  # innaccessible
                       ["时无法打开", "parked"],  # cant open
                       # ["注册", "parked"],  # registered
                       ["不可用", "parked"],  # unavailable
                       ["无误", "parked"],  # incorrect
                       ["所有权", "parked", [1, 2, 3]],  # ownership
                       ["家主", "sale", [1, 2, 3]],  # owner 购买
                       ["所有者", "sale", [1, 2, 3]],  # owner
                       ["过期", "parked"],  # expired
                       # ["建设", "construction"],  # construction
                       ["^网站正在建设中{0,1}$", "construction"],  # under construction
                       ["施工", "construction"],  # construction
                       ["托管", "parked"],  # hosting
                       ["时发布", "parked"],  # hosting
                       ["转让", "parked"],  # transfer
                       ["无对", "parked"],  # incorrect
                       ["没有找到", "parked"],  # cannot find (site)
                       ["中绑定了", "parked"],  # check binding
                       ["过期", "parked"],
                       ["停放", "parked"]]}
DICO_PATTERN_RU = {"core": ["домен", "страниц", "сайт"],
                   "attributes": [["заказать", "order"],
                                  ["продажа", "sale"],
                                  ["припарковал", "parked"],  # parked
                                  ["владел", "parked", [1, 2]],  # owner
                                  ["хостин", "parked"],  # hosting
                                  ["(?<!без )регистра", "parked", [1, 2]],  # register  except if "without" before
                                  ["имени", "parked"],  # named
                                  ["купить", "sale", [1, 2]],  # buy
                                  ["конструкт", "parked"],  # construction
                                  ["хостинг", "parked"],  #
                                  ["истек", "expired"],  #
                                  ["недоступен", "parked"],  # not available
                                  ["прода", "sale", [1, 2]],  # for sale
                                  ["покуп", "sale"],  # purchase
                                  ["зарегистрирова", "parked", [1, 2, 3, 4]],  # registered
                                  ["Создайте", "parked"],  # create
                                  ["отключен", "parked"],  # disabled
                                  ["заблокиров", "parked"],  # blocked
                                  ["Перенаправл", "parked", [1, 2]]  # redirect #
                                  ]}
DICO_PATTERN_SV = {"core": ["( |^)domän( |$|en|namnet|\.)", "( |^)sida( |$|n|\.)", "( |^)hemsida( |$|\.)",
                            "( |^)webbplats( |$|\.)",
                            "( |^)siten( |$|\.)"],
                   "attributes": [
                       ["ägare", "owner"],  # ägare
                       ["registri*er", "registered", [1, 2, 3]],
                       # registrerat, registrering, registrera, registreringen, registriert
                       ["webbhotell", "web hosting"],
                       ["uppbyggnad", "construction"],
                       ["parker[ai]", "parked"],  # parkerat, parkerad, parkering
                       ["konstruktion", "construction"],
                       ["äg(s|er)", "owned"],
                       ["köp[ae]*", "buy"],  # köper, köpa
                       ["salu", "sale", [1, 2]],
                       ["hosting", "hosting"],
                       ["egna", "own"],
                       ["Beställ", "buy"],
                       ["underhåll", "maintenance"],
                       ["Aktivera", "activate", [1, 2, 3, 4, 5, 6]],
                       ["ladda[ts]* upp", "upload"],
                       ["inte (|har )publicera", "not published"],
                       ["ingen sida", "no page"],
                       ["Denna sida visas", "special page", [1, 2, 3, 4, 5, 6]],
                       ["inte öppnat", "not opened"],
                       ["Förfrågan", "not opened"],
                       ["Pris", "price"],
                   ]}
DICO_PATTERN_DE = {"core": ["( |^)domain( |$|\.)", "( |^)seite( |$|\.)", "( |^)webseite( |$|\.)"],
                   "attributes": [
                       ["registriert", "registered"],
                       ["automatisch erstellt", "created automatically", [0]],
                       ["verkauf", "sale"],
                       ["kaufen", "sale"],
                       ["unkonfiguriert", "unconfigured"],
                       ["Aufbau", "construction"],
                       ["Inhaber", "owner", [1, 2, 3, 4]],
                       ["erreichbar", "expired"],
                       ["weitergeleitet", "passed"],
                       ["erwerben", "purchase"],
                       ["keine Website", "no website", [1, 2, 3, 4]],
                       ["geparkt", "parked"],
                       ["zugewiesen", "assigned", [1, 2, 3, 4]],
                       ["Anschaffung", "purchase"],
                       ["Wartungsarbeiten", "maintenance"],
                       ["eingestellt", "discontinued"],
                   ]}
DICO_PATTERN_ET = {
    "core": ["( |^)domeen( |$|\.)", "( |^)lehel( |$|\.)", "( |^)veebilehel( |$|\.)", "( |^)veebileht( |$|\.)"],
    "attributes": [
        ["registreeri", "registered"],
        ["Pargitud", "parked"],
        ["ole veel valmis", "not ready", [0, 4, 5]],
        ["tühjal", "empty", [0, 4, 5]],
        ["müüki", "sale"],
        ["osta", "buy"],
        ["veel üles seatud", "not set up yet"],
        ["suletud lepingu", "expired"],
        ["registriert", "registered"],
    ]}
DICO_PATTERN_IT = {"core": ["( |^)domini", "( |^)sito( |$|\.)", "pagina"],
                   "attributes": [
                       ["configur[ao]", "p1", [1, 2]],
                       ["vend[ui]t[aoi]", "p2", [1, 2], ["noleggio"]],
                       ["acquist[ao]", "p3"],
                       ["registr[ao]", "p4", [1, 2]],
                       # ["registr[ao]", "p4", [1, 2, 3, 4]],
                       ["disponibil", "p5", [], ["cookie", "materiali"]],
                       ["compra", "p6"],
                       ["costruzione", "p7"],
                       ["occupato", "p8", [1, 2, 3, 4]],
                       ["manutenzione", "p9"],
                       ["propriet[àa]", "p10", [1, 2, 3, 4]],
                       ["aggiornamento", "p11", [0, 2]],
                       ["allestimento", "p12"],
                       ["attivo", "p13", [1, 2, 3, 4]],
                       ["Hosting", "p14", [1, 2, 3, 4]],
                       ["generata", "p15", [0, 1, 3, 4]],
                       ["cessione", "p16"],
                       ["prezzo", "p17"],
                       ["presto online", "p18", [0, 2, 3, 4]],
                       ["investire", "p19", [1, 2, 3, 4]],
                       ["trasferimento", "p20", [1, 2, 3, 4]],
                       ["acquire", "p21"],

                   ]}
DICO_PATTERN_DA = {"core": ["domæne", "websted", "( |^)side( |$|\.)"],  # Domænet very often used
                   "attributes": [
                       ["Registrer", "p1", [1]],
                       ["registrator", "p2"],
                       ["Webhosting", "p3", [1, 2, 3, 4]],
                       ["hoste[dst]", "p4", [1]],
                       ["oprettelse", "p5", [1, 2, 3, 4]],
                       ["webhotel", "p6", [1, 2, 3, 4]],
                       ["parker", "p7"],
                       ["ligger", "p8", [1, 2, 3, 4]],
                       ["lukket", "p9", [1, 2, 3, 4]],
                       ["Køb", "p10", [1]],
                       ["ikke e*r* *aktiv", "p11"],
                       ["vælge", "p12", [1, 2, 3, 4]],
                       ["salg", "p13", [1, 2]],
                       ["inaktiv", "p14"],
                       ["konstruktion", "p15"],
                       ["o[pm]bygning", "p16"],
                       ["ikke t*a*g*e*t* *i brug", "p17", [1, 2, 3, 4]],
                       ["hosting", "p18", [1]],
                       ["pris( |$|\.)", "p19", [1, 2, 3, 4]],
                       ["( |^|\.)ejeren*", "p20", [1, 2, 3, 4]],
                       ["vedligehold", "p21", [0]]
                   ]}
DICO_PATTERN_SK = {"core": ["( |^)domén[aouy]", "stránky"],
                   "attributes": [
                       ["registr", "p1"],
                       ["kúpa", "p2"],
                       ["hosting", "p3"],
                       ["dostupnos", "p4", [1, 2, 3]],
                       ["( |^|\.)majit", "p5"],
                       ["Exspirované", "p6"],
                       ["konštrukcii", "p7"],
                       ["parkovaná", "p8"],
                       ["vytvort", "p9", [0, 2, 3]],
                       ["projektom", "p10", [1, 2, 3]],
                       ["predaj", "p11"],
                       ["aktivovaná", "p12", [1]],
                       ["pozastavená", "p13"],
                       ["pr[ií]prav", "p14", [0]],
                       ["výstavbe", "p15", [0]],
                       ["hosťovaná", "p16"],
                       ["záujem", "p16", [1, 2, 3]],
                       ["žiaden obsah", "p17", [0]]
                   ]}

DICO_PATTERN_DEFAULT = {"core": ["", "", ""],
                        "attributes": []}

PATTERN_PARKED = {
    "en": DICO_PATTERN_ENG,
    "fr": DICO_PATTERN_FR,
    "es": DICO_PATTERN_ES,
    "zh-cn": DICO_PATTERN_CN,
    "zh-tw": DICO_PATTERN_CN,
    "ru": DICO_PATTERN_RU,
    "sv": DICO_PATTERN_SV,
    "de": DICO_PATTERN_DE,
    "et": DICO_PATTERN_ET,
    "it": DICO_PATTERN_IT,
    "da": DICO_PATTERN_DA,
    "sk": DICO_PATTERN_SK
}

DICO_CHARACTERS_BY_LGG = {
    "ru": "[a-z '’\"\u0400-\u04FF-]",
    "fr": "[a-z àâäôéèëêïîçùûüÿæœ'’\"-]",
    "zh-cn": "['’\"\u4e00-\u9fff -]",
    "zh-tw": "['’\"\u4e00-\u9fff -]",
    "es": "[a-z áéíñóúü'’\"-]",
    "sv": "[a-z äöå'’\"-]",
    "de": "[a-z äöüß'’\"-]",
    "et": "[a-z äöüß'’\"öšž-]",
    "it": "[a-z '’\"àèéìíîòóùú-]",
    "da": "[a-z '’\"æøå-]",
    "sk": "[a-z '’\"áéíĺóŕúýčďľňšťžäô-]",
    "en": "[a-z '’\"-]",
    "other": "[a-z '’\"-]",
}


def get_unicode_set(lg):
    """ Returns the regular expression of all characters of language lg
    for examples, cf https://www.rexegg.com/regex-interesting-character-classes.html"""
    if lg in DICO_CHARACTERS_BY_LGG:
        return DICO_CHARACTERS_BY_LGG[lg]
    else:
        return DICO_CHARACTERS_BY_LGG["other"]
