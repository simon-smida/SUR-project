# Zadání projektu do SUR 2023/2024

**Bodové ohodnocení:** 25 bodů

## Úkol

Natrénovat detektor jedné osoby z obrázku obličeje a hlasové nahrávky. 

Trénovací vzory jsou k dispozici v archívu zde: [trénovací data](https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2023-2024/SUR_projekt2023-2024.zip)

### Obsah archívu
- `target_train`, `target_dev`: 
  - Trénovací vzory pro detekovanou osobu ve formátu PNG a WAV
- `non_target_train`, `non_target_dev`: 
  - Negativní příklady povolené pro trénování detektoru.

Data můžete rozdělit pro trénování a vyhodnocování úspěšnosti vyvíjeného detektoru, ale rozdělení není závazné. Jméno každého souboru je rozděleno do polí pomocí podtržítek, kde první pole (např. f401) je identifikátor osoby a druhé pole je číslo nahrávacího sezení (01).

### Pravidla pro trénink
- **Není povoleno** využití jiných externích řečových či obrázkových dat nebo již předtrénovaných modelů.
- Možnost augmentace dat (přidáním šumu, rotací, posunutím, změnou velikosti obrázků, změnou rychlosti řeči atd.).

## Vypracování
Ostrá data budou k dispozici 20. dubna ráno. Výsledky musíte uploadovat do WISu do 22. dubna 23:59. Soubor s výsledky bude ASCII se třemi poli na řádku oddělenými mezerou, obsahující:
- Jméno segmentu (jméno souboru bez přípony `.wav` či `.png`)
- Číselné skóre
- Tvrdé rozhodnutí (`1` pro hledanou osobu, jinak `0`)

## Implementace
Detektor můžete implementovat v libovolném programovacím jazyce. Odevzdat můžete několik souborů s výsledky (např. pro systémy rozhodujícím se pouze na základě řečové nahrávky či pouze obrázku). Maximálně bude zpracováno 5 takových souborů.

Soubory s výsledky a zdrojové kódy uploadněte zabalené do jednoho ZIP archívu. Archív bude obsahovat také dokumentaci.pdf popisující vaše řešení a umožní reprodukci vaší práce.

## Inspirace
[Demo příklady pro předmět SUR](https://www.fit.vutbr.cz/study/courses/SUR/public/prednasky/demos/)

Příklady:
- Detekce pohlaví z řeči: `demo_genderID.py`
- Funkce pro načítání PNG souborů (`png2fea`) a extrakci MFCC příznaků z WAV souborů (`wav16khz2mfcc`).

## Hodnocení
- Vše je odevzdáno a nějakým způsobem pracuje: plný počet 25 bodů.
- Pokud něco z výše uvedeného není splněno, bude uděleno méně bodů.

**Poslední modifikace:** 3. dubna 2024, Lukáš Burget
