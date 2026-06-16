"""
新闻分析模块 - 识别新闻对哪些期货品种有影响
支持中英文关键词匹配
"""
from typing import List, Dict, Tuple

# 品种关键词映射表 (中英文)
CONTRACT_KEYWORDS = {
    "RB0": ["螺纹钢", "螺纹", "钢筋", "钢材", "房地产", "基建", "rebar", "steel"],
    "HC0": ["热卷", "热轧卷板", "钢板"],
    "I0": ["铁矿石", "铁矿", "矿石", "iron ore"],
    "J0": ["焦炭", "焦化", "coke"],
    "JM0": ["焦煤", "coking coal"],
    "SS0": ["不锈钢", "stainless"],
    "CU0": ["铜", "沪铜", "电解铜", "精炼铜", "copper"],
    "AL0": ["铝", "沪铝", "电解铝", "氧化铝", "aluminum", "aluminium"],
    "ZN0": ["锌", "沪锌", "zinc"],
    "PB0": ["铅", "沪铅", "lead"],
    "NI0": ["镍", "沪镍", "nickel"],
    "SN0": ["锡", "沪锡", "tin"],
    "AO0": ["氧化铝"],
    "AU0": ["黄金", "金价", "金条", "美联储", "加息", "降息", "央行", "gold", "fed"],
    "AG0": ["白银", "银价", "silver"],
    "SC0": ["原油", "石油", "汽油", "柴油", "OPEC", "欧佩克", "能源", "油价", "伊朗",
            "沙特", "俄罗斯石油", "oil", "crude", "petroleum", "brent", "wti"],
    "FU0": ["燃料油", "fuel oil"],
    "LU0": ["低硫燃油", "低硫"],
    "BU0": ["沥青", "asphalt"],
    "NR0": ["20号胶"],
    "RU0": ["天然橡胶", "橡胶", "rubber"],
    "MA0": ["甲醇", "methanol"],
    "TA0": ["PTA", "精对苯二甲酸"],
    "EG0": ["乙二醇", "glycol"],
    "PF0": ["短纤"],
    "PR0": ["瓶片"],
    "PX0": ["对二甲苯"],
    "PL0": ["丙烯", "propylene"],
    "PP0": ["聚丙烯", "PP"],
    "L0": ["塑料", "聚乙烯", "plastic"],
    "V0": ["PVC", "聚氯乙烯"],
    "EB0": ["苯乙烯", "styrene"],
    "SA0": ["纯碱", "苏打", "soda ash"],
    "SH0": ["烧碱"],
    "UR0": ["尿素", "urea"],
    "FG0": ["玻璃", "glass"],
    "M0": ["豆粕", "豆柏", "饲料", "soybean meal"],
    "Y0": ["豆油", "soybean oil"],
    "P0": ["棕榈油", "棕油", "palm oil"],
    "OI0": ["菜油", "菜籽油", "rapeseed oil"],
    "RM0": ["菜粕", "rapeseed meal"],
    "C0": ["玉米", "corn", "maize"],
    "CS0": ["淀粉", "玉米淀粉", "starch"],
    "A0": ["豆一", "大豆", "soybean"],
    "B0": ["豆二"],
    "CF0": ["棉花", "棉价", "cotton"],
    "SR0": ["白糖", "糖", "sugar"],
    "AP0": ["苹果", "apple"],
    "CJ0": ["红枣", "jujube"],
    "JD0": ["鸡蛋", "egg"],
    "LH0": ["生猪", "猪价", "猪肉", "pig", "hog"],
    "PG0": ["液化气", "LPG"],
    "SI0": ["工业硅", "多晶硅", "silicon"],
    "LC0": ["碳酸锂", "锂", "lithium"],
    "EC0": ["集运", "欧线", "航运", "集装箱", "波罗的海", "运费", "运价", "shipping", "freight", "baltic"],
    "IF0": ["沪深300", "股指", "A股", "股市", "大盘", "上证", "创业板", "科创板",
            "牛市", "熊市", "股票", "IPO", "上市", "stocks", "equity"],
}

# 行业分类
SECTORS = {
    "Black": ["RB0", "HC0", "I0", "J0", "JM0", "SS0"],
    "Non-ferrous": ["CU0", "AL0", "ZN0", "PB0", "NI0", "SN0", "AO0"],
    "Precious": ["AU0", "AG0"],
    "Energy": ["SC0", "FU0", "LU0", "BU0", "NR0", "RU0"],
    "Chemicals": ["MA0", "TA0", "EG0", "PF0", "PR0", "PX0", "PL0", "PP0", "L0",
                   "V0", "EB0", "SA0", "SH0", "UR0", "FG0"],
    "Oils": ["M0", "Y0", "P0", "OI0", "RM0"],
    "Agriculture": ["C0", "CS0", "A0", "B0", "CF0", "SR0", "AP0", "CJ0", "JD0", "LH0", "PG0"],
    "NewEnergy": ["SI0", "LC0"],
    "Shipping": ["EC0"],
    "Index": ["IF0"],
}

SECTOR_CN = {
    "Black": "黑色系", "Non-ferrous": "有色", "Precious": "贵金属",
    "Energy": "能源", "Chemicals": "化工", "Oils": "油脂",
    "Agriculture": "农产品", "NewEnergy": "新能源", "Shipping": "航运", "Index": "股指",
}


def analyze_news_impact(headlines: List[str]) -> List[Dict]:
    """分析新闻头条, 识别影响哪些品种"""
    seen = set()
    results = []

    for headline in headlines:
        if not headline or headline in seen:
            continue
        seen.add(headline)

        hl = headline.lower()
        matched_symbols = set()
        matched_sectors = set()

        for symbol, keywords in CONTRACT_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in hl:
                    matched_symbols.add(symbol)
                    break

        if not matched_symbols:
            continue

        for sector, syms in SECTORS.items():
            if matched_symbols & set(syms):
                matched_sectors.add(sector)

        results.append({
            "headline": headline[:150],
            "symbols": list(matched_symbols)[:6],
            "sectors": [SECTOR_CN.get(s, s) for s in matched_sectors],
            "sector_en": list(matched_sectors)[:3],
            "match_count": len(matched_symbols),
        })

    results.sort(key=lambda x: x["match_count"], reverse=True)
    return results
