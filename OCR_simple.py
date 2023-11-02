# coding=utf-8
from argparse import ArgumentParser
from cv2 import imread,namedWindow,imshow,WINDOW_FREERATIO,waitKey

if __name__=='__main__':
     parser = ArgumentParser()
     parser.add_argument('--path', type=str, default='SharedScreenshot.png', help='目标文件路径')
     parser.add_argument('--module',type=str,default='cnocr',help='识别模块(默认cnocr/tesseract/PaddleOCR)')
     parser.add_argument('--mod',type=str,default='naive_det',help='识别模型(详见对应模块文档)')
     parser.add_argument('--device',type=str,default='cuda',help='cnocr设备(默认cuda/cpu)')
     parser.add_argument('--show',type=str,default='N',help='是否显示图片(Y/N)')
     parser = parser.parse_args()
     if parser.module=='cnocr':
          from cnocr import CnOcr
          if parser.show=='Y':
               a=imread(parser.path)
               namedWindow("final",WINDOW_FREERATIO)
               imshow("final",a)
               waitKey()
          ocr = CnOcr(det_model_name=parser.mod,context=parser.device)
          out = ocr.ocr(parser.path)
          for result in range(len(out)):
               print(out[result]["text"],end='')
               if result+1<len(out) and out[result+1]["text"][0]==' ':
                    print()
     elif parser.module=='PaddleOCR':
          from paddleocr import PaddleOCR
          ocr = PaddleOCR(lang="ch")
          results = ocr.ocr(parser.path)
          if parser.show=='Y':
               a=imread(parser.path)
               namedWindow("final",WINDOW_FREERATIO)
               imshow("final",a)
               waitKey()
          for result in range(len(results[0])):
               print(results[0][result][-1][0],end='')
               if result+1<len(results[0]) and results[0][result+1][-1][0][0]==' ':
                    print()
     elif parser.module == 'tesseract':
          from pytesseract import image_to_string
          a=imread(parser.path)
          if parser.show=='Y':
               namedWindow("final",WINDOW_FREERATIO)
               imshow("final",a)
          print(image_to_string(a, lang=parser.mod if parser.mod != 'naive_det' else 'chi_sim+eng'))
          waitKey()
     else:
          print("请重新输入module")