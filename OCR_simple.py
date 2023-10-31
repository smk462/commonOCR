# coding=utf-8
from argparse import ArgumentParser
from cv2 import imread,namedWindow,imshow,WINDOW_FREERATIO,waitKey
from cnocr import CnOcr
from pytesseract import image_to_string

if __name__=='__main__':
     parser = ArgumentParser()
     parser.add_argument('--path', type=str, default='SharedScreenshot.png', help='目标文件路径')
     parser.add_argument('--module',type=str,default='cnocr',help='识别模块(默认cnocr/tesseract)')
     parser.add_argument('--mod',type=str,default='naive_det',help='cnocr模型')
     parser.add_argument('--device',type=str,default='cuda',help='cnocr设备(默认cuda/cpu)')
     if parser.parse_args().module=='cnocr':
          ocr = CnOcr(det_model_name=parser.parse_args().mod,context=parser.parse_args().device)
          out = ocr.ocr(parser.parse_args().path)
          for result in out:
               print(result["text"])
     else:
          a=imread(parser.parse_args().path)
          namedWindow("final",WINDOW_FREERATIO)
          imshow("final",a)
          print(image_to_string(a, lang='chi_sim+eng')) #tesseract -l chi_sim+eng [parser.parse_args().path] stdout
          waitKey()