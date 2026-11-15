from dataclasses import asdict
from PIL import Image,ImageChops
from ai_meme_layout import place_around_subjects
from meme_project import TextLayer,render_layers


def test_portrait_prefers_readable_block_to_small_or_tall_column():
    image=Image.new('RGB',(1080,1920),'#3b2c20')
    layer=TextLayer(text='Sorry I am late, traffic was brutal and I had to nail this solo.',
                    x=.2,y=.8,width=.3,size=16,font='DejaVuSerif-Bold.ttf')
    original=asdict(layer)
    regions=[dict(label='Drummer',x=.2,y=.3,width=.3,height=.4),
             dict(label='Bandana',x=.5,y=.3,width=.3,height=.4),
             dict(label='Drumsticks',x=.7,y=.3,width=.3,height=.4)]
    placed=place_around_subjects(image,[layer],regions)
    assert placed[0].size>=24 and placed[0].width>=.65
    mask=render_layers(Image.new('RGBA',image.size),placed)
    bounds=mask.getbbox()
    assert bounds and bounds[3]-bounds[1]<image.height*.15
    for region in regions:
        box=(round(region['x']*image.width),round(region['y']*image.height),
             round((region['x']+region['width'])*image.width),round((region['y']+region['height'])*image.height))
        assert mask.crop(box).getbbox() is None
    assert asdict(layer)==original
    # Manual rendering still honors the user's smaller size.
    assert render_layers(image,[layer]).tobytes()!=render_layers(image,placed).tobytes()
