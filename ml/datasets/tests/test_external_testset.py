import pytest

from ml.datasets.build_external_testset import classify_member


@pytest.mark.parametrize("member,expected", [
    ("Mango Dataset/Processed/Anthracnose/img1.jpg", ("Anthracnose", False)),
    ("Mango Dataset/Processed/Sooty Mould/x.JPG", ("Sooty Mold", False)),
    ("Mango Dataset/Processed/Bacterial_Canker/x.png", ("Bacterial Canker", False)),
    ("Mango Dataset/Augmented/Healthy/aug_1.jpg", ("Healthy", True)),
    ("Mango Dataset/Processed/Die Back/x.jpg", (None, False)),
    ("Mango Dataset/readme.txt", (None, None)),
])
def test_classify_member(member, expected):
    assert classify_member(member) == expected


@pytest.mark.parametrize("member,expected", [
    ("Mango Dataset/process data.zip/Sooty Mould/a.jpg", ("Sooty Mold", False)),
    ("Mango Dataset/Augmented Image.zip/Anthracnose/a.jpg", ("Anthracnose", True)),
    ("Mango Dataset/process data.zip/die back/a.jpg", (None, False)),
])
def test_classify_member_real_release_layout(member, expected):
    # layout of the actual 7ghdbftp54 release: two inner zips (see iter_images)
    assert classify_member(member) == expected


def test_iter_images_descends_into_inner_zips(tmp_path):
    import io
    import zipfile
    from ml.datasets.build_external_testset import iter_images

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("Healthy/a.jpg", b"A")
    outer = tmp_path / "o.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("D/process data.zip", inner.getvalue())
        z.writestr("D/readme.txt", b"R")
    with zipfile.ZipFile(outer) as z:
        assert sorted(iter_images(z)) == [("D/process data.zip/Healthy/a.jpg", b"A"), ("D/readme.txt", b"R")]
