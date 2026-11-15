from dataclasses import replace
from PIL import Image, ImageDraw
from ai_meme_layout import composition_hints
from meme_project import TextLayer, render_layers


def test_quiet_region_and_contrast_are_measured_without_changing_background():
    image = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(image)
    for x in range(260, 512, 4):
        draw.line((x, 0, x, 512), fill='black', width=2)
    before = image.tobytes()
    hints = composition_hints(image)
    assert hints[0]['x'] == .05 and hints[0]['width'] == .42
    assert hints[0]['contrast_color'] == 'black'
    assert image.tobytes() == before
    assert all(h['x']+h['width']<=1 and h['y']+h['height']<=.85 for h in hints)
    assert composition_hints(Image.new('RGB',(32,64),'black'))[0]['contrast_color']=='white'


def test_requested_fonts_produce_distinct_pixels_without_font_map():
    background = Image.new('RGB',(512,512),'gray')
    layer = TextLayer(text='A very serious joke',size=36)
    bold = render_layers(background,[layer]).tobytes()
    serif = render_layers(background,[replace(layer,font='DejaVuSerif-Bold.ttf')]).tobytes()
    mono = render_layers(background,[replace(layer,font='DejaVuSansMono-Bold.ttf')]).tobytes()
    assert len({bold,serif,mono}) == 3
    assert render_layers(background,[replace(layer,font='missing-font.ttf')]).tobytes()==bold


def test_narrow_ai_caption_does_not_fill_the_image_height():
    from ai_meme_layout import fit_caption_layers
    background=Image.new('RGB',(770,378),'gray')
    layer=TextLayer(text="Music's cheaper than a therapist...and less judgment.",
                    x=.6,y=.05,width=.3,size=31,font='DejaVuSerif-Bold.ttf')
    fitted=fit_caption_layers(background,[layer])[0]
    from PIL import ImageChops
    bounds=ImageChops.difference(render_layers(background,[fitted]).convert('RGB'),background).getbbox()
    assert bounds is not None and bounds[3]-bounds[1] <= round(background.height*.25)
    assert fitted.text==layer.text and fitted.font==layer.font and fitted.size<layer.size
