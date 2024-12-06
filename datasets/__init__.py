import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import OneHotEncoder
from os import listdir, mkdir
from os.path import exists



font_size=96

threshold=100

# 部首字
partial='一丨丶丿乙亅二亠人儿入八冂冖冫几凵刀力勹匕匚匸十卜卩厂厶又口囗土士夂夊夕大女子宀寸小尢尸屮山巛工己巾干幺广廴廾弋弓彐彡彳心戈戶手支攴文斗斤方无日曰月木欠止歹殳毋比毛氏气水火爪父爻爿片牙牛犬玄玉瓜瓦甘生用田疋疒癶白皮皿目矛矢石示禸禾穴立竹米糸缶网羊羽老而耒耳聿肉臣自至臼舌舛舟艮色艸虍虫血行衣襾見角言谷豆豕豸貝赤走足身車辛辰辵邑酉釆里金長門阜隶隹雨青非面革韋韭音頁風飛食首香馬骨高髟鬥鬯鬲鬼魚鳥鹵鹿麥麻黃黍黑黹黽鼎鼓鼠鼻齊齒龍龜龠'
# 千字帖(去掉部首字)
thousand='天地宇宙洪荒盈昃宿列張寒來暑往秋收冬藏閏餘成歲律召調陽雲騰致露結爲霜麗出崑岡劍號巨闕珠稱夜光果珍李柰菜重芥薑海鹹河淡鱗潛翔師帝官皇始制字乃服裳推位讓國有虞陶唐弔民伐罪周發殷湯坐朝問道垂拱平章愛育黎伏戎羌遐邇壹體率賓歸王鳴鳳在樹駒場化被草賴及萬蓋此髮四五常恭惟鞠養豈敢毀傷慕貞絜男效才良知過必改得能莫忘罔談彼短靡恃信使可覆器欲難量墨悲絲淬詩讚羔景維賢克念作聖德建名形端表正空傳聲虛堂習聽禍因惡積福緣善慶尺璧寶陰是競資事君嚴與敬孝當竭忠則盡命臨深履薄夙興溫清似蘭斯馨如松之盛川流不息淵澄取映容若思辭安定篤初誠美慎終宜令榮業所基籍甚無竟學優登仕攝職從政存以棠去益詠樂殊貴賤禮別尊卑上和下睦夫唱婦隨外受傅訓奉母儀諸姑伯叔猶兒孔懷兄弟同氣連枝交友投分切磨箴規仁慈隱惻造次弗離節義廉退顛沛匪虧性靜情逸動神疲守眞志滿逐物意移堅持雅操好爵縻都華夏東西京背邙洛浮渭據涇宮殿盤鬱樓觀驚圖寫禽獸畫彩仙靈丙舍傍啟甲帳對楹肆筵設席瑟吹笙升階納陛弁轉疑星右通廣內左達承明既集墳典亦聚群英杜稾鍾隸漆書壁經府羅將相路俠槐卿封縣家給千兵冠陪輦驅轂振纓世祿侈富駕肥輕策功茂實勒碑刻銘磻溪伊尹佐時阿衡奄宅曲微旦孰營桓公匡合濟弱扶傾綺迴漢惠說感武丁俊乂密勿多寔寧晉楚更霸趙魏困橫假途滅虢踐會盟何遵約法韓弊煩刑起翦頗牧軍最精宣威沙漠馳譽丹九州禹跡百郡秦并嶽宗恆岱禪主云亭雁紫塞雞城昆池碣鉅野洞庭曠遠緜邈巖岫杳冥治本於農務茲稼穡俶載南畝我藝稷稅熟貢新勸賞黜陟孟軻敦素史秉直庶幾中庸勞謙謹敕聆察理鑑貌辨貽厥嘉猷勉其祗植省躬譏誡寵增抗極殆辱近恥林皋幸即兩疏機解組誰逼索居閒處沈默寂寥求古尋論散慮逍遙欣奏累遣慼謝歡招渠荷的歷園莽抽條枇杷晚翠梧桐早凋陳根委翳落葉飄颻游鵾獨運凌摩絳霄耽讀翫市寓囊箱易輶攸畏屬垣牆具膳餐飯適充腸飽飫烹宰飢厭糟糠親戚故舊少異糧妾御績紡侍帷房紈扇圓潔銀燭煒煌晝眠寐籃筍象牀弦歌酒讌接杯舉觴矯頓悅豫且康嫡後嗣續祭祀烝嘗稽顙再拜悚懼恐惶箋牒簡要顧答審詳骸垢想浴執熱願涼驢騾犢特駭躍超驤誅斬賊盜捕獲叛亡布射遼丸嵇琴阮嘯恬筆倫紙鈞巧任釣釋紛利俗並皆佳妙施淑姿顰妍笑年每催曦暉朗耀琁璣懸斡晦魄環照指薪脩祜永綏吉劭矩步引領俯仰廊廟束帶矜莊徘徊瞻眺孤陋寡聞愚蒙等誚謂語助者焉哉乎也'


img_height = 96  # image height
img_width = 96  # image width
img_shape = (img_height, img_width)

def loadImage(path):
    # print(path)
    img = Image.open(path)
    img = img.resize(img_shape).convert('L')
    img = np.array(img) / 255.0
    # print(img.shape)
    img = np.expand_dims(img, -1) # (img_height, img_width, 1)
    return img

def loadImagesInFolder(folderPath):
    print(folderPath)
    imgNames = listdir(folderPath)
    images = np.empty((0, img_width, img_height, 1))
    for imgName in imgNames:
        imgPath = folderPath + imgName
        # Use you favourite library to load the image
        image = loadImage(imgPath)
        images = np.concatenate([images,np.expand_dims(image, axis=0)], axis=0)
    return images

def loadDataset(path):
    print(path)
    folderNames = listdir(path)
    images = []
    labels = []
    for label, folderName in enumerate(folderNames):
        if(folderName == 'FontData'):
            continue
        if(folderName == '.git'):
            continue
        if(folderName == 'README.md'):
            continue
        folderPath = path + folderName + "/"
        folderImages = loadImagesInFolder(folderPath)
        images.append(folderImages)
        labels.append(np.full([len(folderImages)], label))
    
    X = np.concatenate(images, axis=0)
    X = np.moveaxis(X, -1, 1)

    Y = np.concatenate(labels, axis=0)
    oneshot_encoder = OneHotEncoder(sparse_output=False)
    Y = oneshot_encoder.fit_transform(np.expand_dims(Y, -1))

    return X, Y

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, device, dtype = torch.float32):
        ## TODO
        self._X = torch.tensor(X, dtype=dtype, device=device)
        self._Y = torch.tensor(Y, dtype=dtype, device=device)

    def __len__(self):
        ## TODO
        return len(self._X)

    def __getitem__(self, idx):
        ## TODO
        return self._X[idx], self._Y[idx]


def extract_font(font, char):
    l, u, r, d= font.getbbox(char)
    # print(l, u, r, d)
    image = Image.new('L', (font_size,font_size))
    draw = ImageDraw.Draw(image)
    draw.text((0, -u+(font_size-(d-u))/2), char, font=font, fill=255)
    # display(image)
    return np.array(image)

def fontImageGenerator(sourcePath, targetPath):

    if(exists(targetPath)):
        return

    # fetch font info
    fontNames = listdir(sourcePath)
    fontPaths = [sourcePath+fontName for fontName in fontNames]
    # print(fontPaths)
    fonts = [ImageFont.truetype(font, font_size) for font in fontPaths]

    # transform into image(train)
    images=[[extract_font(font,char)>threshold for char in partial]for font in fonts]

    # save images
    mkdir(targetPath)
    mkdir(targetPath + 'partial/')
    for i, fontImages in enumerate(images):
        font = fontNames[i][:-4]
        font = targetPath + 'partial/' + font
        mkdir(font)
        # print(font)
        for j, charImage in enumerate(fontImages):
            Image.fromarray(charImage).save(f"{font}/{partial[j]}.jpg")
            print(f"{font}/{partial[j]}.jpg")
        
    
    # transform into image(test)
    images=[[extract_font(font,char)>threshold for char in thousand]for font in fonts]

    # save images
    mkdir(targetPath + 'thousand/')
    for i, fontImages in enumerate(images):
        font = fontNames[i][:-4]
        font = targetPath + 'thousand/' + font
        mkdir(font)
        # print(font)
        for j, charImage in enumerate(fontImages):
            Image.fromarray(charImage).save(f"{font}/{thousand[j]}.jpg")
            print(f"{font}/{thousand[j]}.jpg")
        

        
    return len(fonts)