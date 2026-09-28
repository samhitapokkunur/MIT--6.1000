from functools import wraps
import pset
import string
import sys
import unittest
from unittest.mock import MagicMock, patch


############################################################
# set up test case examples
############################################################


considered_punctuation = string.punctuation
alphabet = (
    string.ascii_lowercase
    + string.ascii_uppercase
    + string.digits
    + considered_punctuation
)

max_initial_shift = 16
magic_number_range = range(4, 16)

three_bears = """
    Once upon a time there were three Bears, who lived together in a
    house of their own, in a wood. One of them was a Little Wee Bear,
    and one was a Middle-sized Bear, and the other was a Great Big Bear.
    And they had each a chair to sit in; a little chair for the Little
    Wee Bear; and a middle-sized chair for the Middle-sized Bear; and a
    great chair for the Great Big Bear. And they had each a bed to sleep
    in; a little bed for the Little Wee Bear; and a middle-sized bed for
    the Middle-sized Bear; and a great bed for the Great Big Bear.
"""

miser = """
    A Miser sold all that he had and bought a lump of gold, which he
    buried in a hole in the ground by the side of an old wall and went
    to look at daily. One of his workmen observed his frequent visits to
    the spot and decided to watch his movements. He soon discovered the
    secret of the hidden treasure, and digging down, came to the lump of
    gold, and stole it. The Miser, on his next visit, found the hole
    empty and began to tear his hair and to make loud cries. A neighbor,
    seeing him overcome with grief and learning the cause, said, "Pray
    do not grieve so; but; go and take a stone, and place it in the
    hole, and fancy that the gold is still lying there. It will do you
    quite the same service; for when the gold was there, you had it not,
    as you did not make the slightest use of it.
"""

tortured_poets = """
    You left your typewriter at my apartment
    Straight from the Tortured Poets Department
    I think some things I never say
    Like, "Who uses typewriters anyway?"
    But you're in self-sabotage mode
    Throwing spikes down on the road
    But I've seen this episode and still loved the show
    Who else decodes you?
    And who's gonna hold you like me?
    And who's gonna know you, if not me?
    I laughed in your face and said
    "You're not Dylan Thomas, I'm not Patti Smith
    This ain't the Chelsea Hotel, we're modern idiots"
    And who's gonna hold you like me?
    Nobody
    No-fucking-body
    Nobody
"""

single_char_encrypted_dict_1 = {
    "a": "c",
    "A": "C",
    "!": "#",
    "1": "3",
}

single_char_encrypted_dict_2 = {
    "a": "j",
    "A": "J",
    "!": "*",
    "1": "!",
}

short_examples_encrypted_dict_1 = {
    "easy": "gcuA",
    "This is a simple test": "Vjku ku c ukorng vguv",
    "Try hyphenated-test case": "VtA jArjgpcvgf/vguv ecug",
    "Without a doubt, 6.1000 is the best subject ever! And the staff just amazing!!": "Ykvjqxw d grxew/ 9;4455 nx ynk hkyA zBirmkB mDmz) Ivl Bpm ABioo sDBC jvjIrwp**",
    "Trying ain't always easy, but she's worth it": "VtAkpi dlq+y grCgEy lhzF= jCB Apm/A EwzBp qC",
    "Cousins' would never be the same": "Eqxvlqw+ Btzqj tkBkx hk znk yhtl",
    "this is lowercase": "vjku ku nqzhufdvh",
    "THIS IS UPPERCASE": "VJKU KU WSSHUFDVH",
    "": "",
    "!,": "#/",
    "÷": "÷",
}

short_examples_encrypted_dict_2 = {
    "easy": "hdvB",
    "This is a simple test": "Wkmw mw e wmqtpi xiwx",
    "Try hyphenated-test case": "WuC lCtlirexih;xiwx gewi",
    "Without a doubt, 6.1000 is the best subject ever! And the staff just amazing!!": "Zlwksyx e hsyfx: !<5444 mw xli fiwx wyfnigx iziv% Esi ymj xyfkl pAyz gsgFotm''",
    "Trying ain't always easy, but she's worth it": "WuCmrk fns,y fqBfDx jfxD; gzy xmj,x Btwym ny",
    "Cousins' would never be the same": "Frxvlqv* zrxog qhyhu fi xli weqi",
    "this is lowercase": "wkmw mw psAivgewi",
    "THIS IS UPPERCASE": "WKLV LV XSSHUFDVH",
    "": "",
    "1234": "4567",
    "!,": "$/",
    "÷": "÷",
}

long_examples_encrypted_dict_1 = {
    three_bears: """\n    Qpeg wrqq e xmqj ynkxk Ckxk znxkk Hkgxy= Epw tqEnm CxpnCqoB sx k\n    ryFDp zq FtqvE BJA] vA n JBBq` 2Bs Ct HvsA KoG p 0xIIAt "tt QtpG`\n    qEu FEv NrJ r 3zuuCv|JzQvu SvrI{ rEu Kyw GMAxK PtL t ZKxtM UBz UxtKa\n    UHx NByT Cvy AwyE x zExFO RM QGS HMj z KHSSKD BGAIR FOS UIF #KVVNG\n    :HH 4IEVo ERH E QMHHPIkWN4KJ INGPY MVY 0OM +RMMUNp1R8NM !NJ0t JWN K\n    Q1OL4 OUNV4 T25 7VS +6TP8 \'XV \'TP6w &2S 8WU% XQT UQSX Q RUT 94 92VV6\n    Z4C S 30""3W UXW Z8" $1Y @2$$5Y aZZ -ZV#G X!0 X 950081D&5-10 Y10 2"%\n    \'42 _722!3F(7/32 ;3Z\'J Z#3 0 6(41+ 365 7\'* ,96 ]*62, >!8 >62*J\n""",
    miser: """\n    C Olvhu vrog doo wkdw kh kdg dqh fsylmy g rAsv ul murj< Cnoio ol\n    iCzqnm rw j qywp ty Ftq tEBHAq pM Hvs Gxst Du pC DAs LpAA pCs LtCI\n    ID ADDz pI spxAN{ 3Ct Du wxH LDGACuD ErIuHLut xyJ wIvHLvEK MzJzKJ KF\n    Kyw KHGM tGx xyxDyAz PK SwPyE EFP JLSBJBKQPe 4C QMML BGRBNUDQDC SGE\n    SECRET OG VJG KLGHIR XVIEWZWKl GUK KPNNPUN KV3Um JHTL 0V 0OM T2UX WO\n    QYVNp KXN 23YVO T4t @TQ /U4R4t 21 VW7 2T#9 "Y8Y9v V4"4V "ZW 074Y\n    Y69$) U7X VY0U7 $8 $YU" 12# 2V3# W!0 \'" 9X71 8"(0 Z%51&E . !1534Z#\'E\n    (337#6 78# &-5)3&$5 .9+9 8*!67 2&5 $62*&!&8 ,96 43.,7I ,3"6I yd,4>\n    8* )*/ "-$9;9 .*P 6:/P "* 5)8 /5&9 5 ./*)9K 5)8 +(68! %: %* :$!\n    $+)"M 7+" $8,![ <&8< <&# &.+$ )= >?*-- -^*/( ?)&=\'T e@ ]+// \'< `<\\\n    >\\,[( [+) [%;) [)@^-\')Y *=[ `-*= ^.+ .@=, })_ {;._/Z c]} <+/ >} ]^}0\n    ,| d^~ />/ ]^} \\,@: }=; }\\?=?a=bc db> |? \\c6\n""",
    tortured_poets: """\n    0qx ohix Csyw yEvkCxozkx gz sE gvgxzsktz\n    Yzxgomnz lxus znk ZuxzAxkj Vukzy KlwhyBumvB\n    Q Bpqwt Bxvn CqsxqD T yqIrE FnL\n    Yvxr] /9uB HFrF GMDsKFwHsFG pCNLpNd;\n    QJI NDJ@Gu yE JvCw|JrsFKrxv DFuv\n    !yJGPBGA MJCEyN yKSJ KJ PDA NKwz\n    XQP 4|SB PBBK QEFP BMFPLAB xKA PQFII ILSBA QEB PELT\n    *EL BIPB ABzLABP VLRl\n    XKA TEL|P DLKKx ELIA VLR IFHB JBl\n    XKA TEL|P DLKKx HKLT VLRc FC KLQ JBl\n    5 IxRDEBA FK VLRO DyBD zMC RzHC\n    _/OVbSF OQV 50NCP ,JQODVi #fR STY +G00P :TP0O\n    <PQ0 IQVi2 2QN "QNU1NJ \'X2NUo 5Nj0N VXMN0W SNSY32f\n    !XN 6RYk2 QYXXK RYVN 8Y4 WUWR ZRB\n    ;1O2R#\n    =4xW#V316ZzU7W)\n    \\8V8X)\n""",
}

long_examples_encrypted_dict_2 = {
    three_bears: """\n    Rqfh xsrq d wlph wkivi Aivi xlvii Fievw: Als pmzih xskjymjw ns f\n    mtzxj tk znkpy vDu> qv i Exxm[ Xwn xo Cqnv FjB j UrCCuo 6oo LokC[\n    lyo zyp HlD l Xtoowp\\DtKpo MplC[ lyo Etq AFtqD ImE n TEsoH Pwu QtpG{\n    QDt JxuP yru wsuA t vAtBK MH LBM BGd t ECNNFy wBuDM AJM ODA 7EQQIB\n    *BB YCyPi yLB y KGBBJCeQGXCB AFyGP DMP RFC !GBBJCeQGXCB ZCyPi yLB y\n    EQDzS BGzHQ EOS UIF 7SFBU 2JH 2FBSi 1OF VJG0 JCF GCEK D EIH XS WPIIT\n    MRo E QNYZRK HKJ LUX ZNK (O00SL >LL 8MIZs IVL I UQLLTMo0Q7ML JML NWZ\n    1PM +QLLTMo0Q7ML 9MIZs IVL I O0NJ2 KNM OY2 4SP (2PL4 #TR #PL2s\n""",
    miser: """\n    D Plvhu wsph epp xlex li leh erh fsykmy f qzru um nvsk> Eqrlq qn\n    kECtpo ty l szwp ty Etq sDAGzp nK Ftq Evqr Bs nA Byq Jnyy nAq JrAG\n    GB yBBx nG qnvyL_ 1Ar Bs uwG KCFyAsB CpGsFJsr vwG tFsEJtCI KxHxIH ID\n    Iwt HEEK rEv vwuBwxw MH PtMvA ABL FHOyGyHNMb 1y MIIH xCMwIPyLyx NBy\n    MywLyN Iz NBy BCxxyH NLyuNPMza wJz zEDDFKD ALTKc zxJB QL QEC JSKN MD\n    ENKCe zMC RSNKD HSg )HE $JTFSg PO JKU PGZV XKUKVh HQWPF VJG JQNG\n    GOSW1 DQG EIKES YU ZKGX NOZ OIQZ IVL 1W UISM TW2L KZQM0p 8 WNRPQKX0o\n    1NNRWP QRV X4N0LXVN 5R2Q P0RNO KXN VOL2YUZS 5TQ ON75Rs 5NVQs i=5O$\n    S3 238 W7YV#V 95B S"!B X5 R4U !R1V R 9!54Vw R4U 62RTV Z! Z4 !YW\n    Z63Wx S5V XS5U\' "ZS" "ZW Y74W 1" "#144 4(16Z #0X!XA <# &144 W7 (7$\n    9$1#X #0X "T5X "X!%1VXD Y8# (2Z8 &30 2!7Z )X& \'41%1C ,"( 4X0 5\' !"\'C\n    X& ,"( 050 !"\' 9X71 \'41 &8645(2\'( )(3 $4 7)H\n""",
    tortured_poets: """\n    1rx ohiw Brxu xCtiAvmxiv ex qC etevxqirx\n    Wxvemkmy kxut Aol 0vyABylk XwmBA LmxizBunwC\n    R Cqrwt Bxvn CqrwpB R wnEoB CkJ\n    Wtvp[ -7sz FDpD EKBqIDuGrEG oBMKoMc:\n    PIH MCI?Fs wB Gszt_GopCHout BDst\n    8xHENzEy KHACwK vGOG HG MAx KHtw\n    UNM 1^Oy MyyH NBCM yJCMIxy uIy NOEII ILSBA QEC QFMU\n    +FM CJQC BCAMBCQ WMSm\n    YMC VGN~R FNMMz GNKC XNT KHJD LDn\n    ZMC VGN~R FNMMz JMNV XNTe HE NOU NFp\n    9 MBVHIFE KP 0QWT HCEH DQG VDLG\n    }<RXdUH QRW 61OES /MTRFXk %gS TUZ ,GZZO :TP0O\n    <PQ0 IQVi1 1PM !PMT0MI &W1MTn 4MiZM UWLMZV QLQW10d\n    8WM 5QXj1 PXWWJ QXUM 7X3 URTN VNx\n    -XKXM7\n    -XpO4MUSXQqLYO!\n    :0N0P!\n""",
}

count_words_dict = {
    'hello world': 2,
    'Harvard is cool, but MIT is better.': 5,
    'I am a dog person; I am also a cat person.': 11,
    'Drink from the fire hose!': 5,
    'How are you?': 3,
    'I am tired.': 3,
    "dog's toy": 2,
    "I don't know": 3,
    '\n Is this computer working?\n Yes': 5,
    '': 0,
    'state-of-the-art': 4,
    ' hi': 1,
    '"My whole life was measured in summers"': 7,
    "A parent/guardian must sign the form": 7,
    }

short_possible_deciphered_txts_dict = {
    "gcuA": ["_[in", ".*`a", ".)_a", "easy", "-)_a", "EASY"],
    "Vjku ku c ukorng vguv": ["This is a simple test", "*HIS IS A SIMPLE TEST"],
    "VtA jArjgpcvgf/vguv ecug": [
        "Try hyphenated-test case",
        "Kip }pg}`e\\k`_$k`jj ][i^",
        "Kip }pg}`e[j_^#j_ij ][i_",
    ],
    "Ykvjqxw d grxew/ 9;4455 nx ynk hkyA zBirmkB mDmz) Ivl Bpm ABioo sDBC jvjIrwp**": [
        "Without a doubt, 6.1000 is the best subject ever! And the staff just amazing!!"
    ],
    "VtAkpi dlq+y grCgEy lhzF= jCB Apm/A EwzBp qC": [
        "Trying ain't always easy, but she's worth it"
    ],
    "Eqxvlqw+ Btzqj tkBkx hk znk yhtl": ["Cousins' would never be the same"],
    "vjku ku nqzhufdvh": [
        "a?@} >{ ?\\a._+(^,",
        "this is loxfsdbtf",
        "a?@~ @~ ]`e=~;/a=",
        "THIS IS LOXFSDBTF",
        "c@[a [a ^{f>~;/a<",
        "this is lowercase",
        "b@[a [a ^{e=~;/a=",
        "THIS IS LOWERCASE",
        "a?@} >{ ?\\a/`,*`.",
        "this is lnwercase",
        "b@[a [a ^`e=~;/a=",
        "THIS IS LNWERCASE",
    ],
    "VJKU KU WSSHUFDVH": ["THIS IS UPPERCASE", "this is uppercase"],
    "": [""],
    "#/": ["IT", "DO", "it", "do", "|i", "@a", "MY", "AM", "my", "am", "{i", "?a"],
    "÷": ["÷"],
}

long_possible_deciphered_txts_dict = {
    "\n    Qpeg wrqq e xmqj ynkxk Ckxk znxkk Hkgxy= Epw tqEnm CxpnCqoB sx k\n    ryFDp zq FtqvE BJA] vA n JBBq` 2Bs Ct HvsA KoG p 0xIIAt \"tt QtpG`\n    qEu FEv NrJ r 3zuuCv|JzQvu SvrI{ rEu Kyw GMAxK PtL t ZKxtM UBz UxtKa\n    UHx NByT Cvy AwyE x zExFO RM QGS HMj z KHSSKD BGAIR FOS UIF #KVVNG\n    :HH 4IEVo ERH E QMHHPIkWN4KJ INGPY MVY 0OM +RMMUNp1R8NM !NJ0t JWN K\n    Q1OL4 OUNV4 T25 7VS +6TP8 'XV 'TP6w &2S 8WU% XQT UQSX Q RUT 94 92VV6\n    Z4C S 30\"\"3W UXW Z8\" $1Y @2$$5Y aZZ -ZV#G X!0 X 950081D&5-10 Y10 2\"%\n    '42 _722!3F(7/32 ;3Z'J Z#3 0 6(41+ 365 7'* ,96 ]*62, >!8 >62*J\n": [
        "\n    Once upon a time there were three Bears, who lived together in a\n    house of their own, in a wood. One of them was a Little Wee Bear,\n    and one was a Middle-sized Bear, and the other was a Great Big Bear.\n    And they had each a chair to sit in; a little chair for the Little\n    Wee Bear; and a middle-sized chair for the Middle-sized Bear; and a\n    great chair for the Great Big Bear. And they had each a bed to sleep\n    in; a little bed for the Little Wee Bear; and a middle-sized bed for\n    the Middle-sized Bear; and a great bed for the Great Big Bear.\n"
    ],
    '\n    C Olvhu vrog doo wkdw kh kdg dqh fsylmy g rAsv ul murj< Cnoio ol\n    iCzqnm rw j qywp ty Ftq tEBHAq pM Hvs Gxst Du pC DAs LpAA pCs LtCI\n    ID ADDz pI spxAN{ 3Ct Du wxH LDGACuD ErIuHLut xyJ wIvHLvEK MzJzKJ KF\n    Kyw KHGM tGx xyxDyAz PK SwPyE EFP JLSBJBKQPe 4C QMML BGRBNUDQDC SGE\n    SECRET OG VJG KLGHIR XVIEWZWKl GUK KPNNPUN KV3Um JHTL 0V 0OM T2UX WO\n    QYVNp KXN 23YVO T4t @TQ /U4R4t 21 VW7 2T#9 "Y8Y9v V4"4V "ZW 074Y\n    Y69$) U7X VY0U7 $8 $YU" 12# 2V3# W!0 \'" 9X71 8"(0 Z%51&E . !1534Z#\'E\n    (337#6 78# &-5)3&$5 .9+9 8*!67 2&5 $62*&!&8 ,96 43.,7I ,3"6I yd,4>\n    8* )*/ "-$9;9 .*P 6:/P "* 5)8 /5&9 5 ./*)9K 5)8 +(68! %: %* :$!\n    $+)"M 7+" $8,![ <&8< <&# &.+$ )= >?*-- -^*/( ?)&=\'T e@ ]+// \'< `<\\\n    >\\,[( [+) [%;) [)@^-\')Y *=[ `-*= ^.+ .@=, })_ {;._/Z c]} <+/ >} ]^}0\n    ,| d^~ />/ ]^} \\,@: }=; }\\?=?a=bc db> |? \\c6\n': [
        '\n    A Miser sold all that he had and bought a lump of gold, which he\n    buried in a hole in the ground by the side of an old wall and went\n    to look at daily. One of his workmen observed his frequent visits to\n    the spot and decided to watch his movements. He soon discovered the\n    secret of the hidden treasure, and digging down, came to the lump of\n    gold, and stole it. The Miser, on his next visit, found the hole\n    empty and began to tear his hair and to make loud cries. A neighbor,\n    seeing him overcome with grief and learning the cause, said, "Pray\n    do not grieve so; but; go and take a stone, and place it in the\n    hole, and fancy that the gold is still lying there. It will do you\n    quite the same service; for when the gold was there, you had it not,\n    as you did not make the slightest use of it.\n'
    ],
    "\n    0qx ohix Csyw yEvkCxozkx gz sE gvgxzsktz\n    Yzxgomnz lxus znk ZuxzAxkj Vukzy KlwhyBumvB\n    Q Bpqwt Bxvn CqsxqD T yqIrE FnL\n    Yvxr] /9uB HFrF GMDsKFwHsFG pCNLpNd;\n    QJI NDJ@Gu yE JvCw|JrsFKrxv DFuv\n    !yJGPBGA MJCEyN yKSJ KJ PDA NKwz\n    XQP 4|SB PBBK QEFP BMFPLAB xKA PQFII ILSBA QEB PELT\n    *EL BIPB ABzLABP VLRl\n    XKA TEL|P DLKKx ELIA VLR IFHB JBl\n    XKA TEL|P DLKKx HKLT VLRc FC KLQ JBl\n    5 IxRDEBA FK VLRO DyBD zMC RzHC\n    _/OVbSF OQV 50NCP ,JQODVi #fR STY +G00P :TP0O\n    <PQ0 IQVi2 2QN \"QNU1NJ 'X2NUo 5Nj0N VXMN0W SNSY32f\n    !XN 6RYk2 QYXXK RYVN 8Y4 WUWR ZRB\n    ;1O2R#\n    =4xW#V316ZzU7W)\n    \\8V8X)\n": [
        "\n    You left your typewriter at my apartment\n    Straight from the Tortured Poets Department\n    I think some things I never say\n    Like, \"Who uses typewriters anyway?\"\n    But you're in self-sabotage mode\n    Throwing spikes down on the road\n    But I've seen this episode and still loved the show\n    Who else decodes you?\n    And who's gonna hold you like me?\n    And who's gonna know you, if not me?\n    I laughed in your face and said\n    \"You're not Dylan Thomas, I'm not Patti Smith\n    This ain't the Chelsea Hotel, we're modern idiots\"\n    And who's gonna hold you like me?\n    Nobody\n    No-fucking-body\n    Nobody\n"
    ],
}


############################################################
# test case settings
############################################################


# DO NOT MODIFY
def case_options(points, failure, error):
    """Decorator to add points and messages to a test case."""

    def decorator(func):
        # directly set attributes on the original function
        func.points = points
        func.failure_message = failure
        func.error_message = error

        @wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(args[-1], MagicMock):
                args = args[:-1]
            return func(*args, **kwargs)

        return wrapper

    return decorator


# DO NOT MODIFY
def testsuite_options(timeout, weight):
    """Decorator to add timeout and weight to a test suite."""

    def decorator(cls):
        # directly set attributes on the original class
        cls.timeout = timeout
        cls.weight = weight
        return cls

    return decorator


############################################################
# part 3.1: decrypt single char
############################################################


@testsuite_options(4, 1)
class TestPart3_1(unittest.TestCase):
    """Test decrypt char"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        self.shift1 = 2
        self.shift2 = 9
        self.shift3 = 0

    @case_options(
        1,
        "Your code does not decrypt char correctly",
        "Task test_decrypt_char_1 error",
    )
    def test_decrypt_char_1(self):
        for char, encrypted_char in single_char_encrypted_dict_1.items():
            actual = pset.decrypt_char(encrypted_char, alphabet, self.shift1)
            self.assertEqual(
                char,
                actual,
                f"Incorrect decrypted content, Expected: {char}, got: {actual}",
            )

    @case_options(
        1,
        "Your code does not decrypt char correctly",
        "Task test_decrypt_char_2 error",
    )
    def test_decrypt_char_2(self):
        for char, encrypted_char in single_char_encrypted_dict_2.items():
            actual = pset.decrypt_char(encrypted_char, alphabet, self.shift2)
            self.assertEqual(
                char,
                actual,
                f"Incorrect decrypted content, Expected: {char}, got: {actual}",
            )

    @case_options(
        1,
        "Your code does not decrypt char correctly",
        "Task test_decrypt_char_3 error",
    )
    def test_decrypt_char_3(self):
        for char, encrypted_char in single_char_encrypted_dict_1.items():
            actual = pset.decrypt_char(encrypted_char, alphabet, self.shift3)
            self.assertEqual(
                encrypted_char,
                actual,
                f"Incorrect decrypted content, Expected: {encrypted_char}, got: {actual}",
            )


############################################################
# part 3.2: encrypt text
############################################################


@testsuite_options(4, 1)
class TestPart3_2(unittest.TestCase):
    """Test Encryption"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        self.shift1 = 2
        self.shift2 = 3
        self.shift3 = 0
        self.magic_number1 = 8
        self.magic_number2 = 10
        self.magic_number3 = 1000

    @case_options(
        1,
        "Your code does not encrypt plaintext correctly",
        "Task test_encrypt_short_1 error",
    )
    def test_encrypt_short_1(self):
        for plaintext, expected_ciphertext in short_examples_encrypted_dict_1.items():
            actual_ciphertext = pset.encrypt(
                plaintext, alphabet, self.shift1, self.magic_number1
            )
            self.assertEqual(
                expected_ciphertext,
                actual_ciphertext,
                f"Incorrect encrypted content. Expected: {expected_ciphertext}, got: {actual_ciphertext}",
            )

    @case_options(
        1,
        "Your code does not encrypt plaintext correctly",
        "Task test_encrypt_short_2 error",
    )
    def test_encrypt_short_2(self):
        for plaintext, expected_ciphertext in short_examples_encrypted_dict_2.items():
            actual_ciphertext = pset.encrypt(
                plaintext, alphabet, self.shift2, self.magic_number2
            )
            self.assertEqual(
                expected_ciphertext,
                actual_ciphertext,
                f"Incorrect encrypted content. Expected: {expected_ciphertext}, got: {actual_ciphertext}",
            )

    @case_options(
        1,
        "Your code does not encrypt plaintext correctly",
        "Task test_encrypt_short_3 error",
    )
    def test_encrypt_short_3(self):
        actual_ciphertext = pset.encrypt("iridescent", alphabet, 0, 1000)
        self.assertEqual(
            "iridescent",
            actual_ciphertext,
            f"Incorrect encrypted content. Expected: iridescent, got: {actual_ciphertext}",
        )

    @case_options(
        1,
        "Your code does not encrypt plaintext correctly",
        "Task test_encrypt_long_1 error",
    )
    def test_encrypt_long_1(self):
        for plaintext, expected_ciphertext in long_examples_encrypted_dict_1.items():
            actual_ciphertext = pset.encrypt(
                plaintext, alphabet, self.shift1, self.magic_number1
            )
            self.assertEqual(
                expected_ciphertext,
                actual_ciphertext,
                f"Incorrect encrypted content. Expected: {expected_ciphertext}, got: {actual_ciphertext}",
            )

    @case_options(
        1,
        "Your code does not encrypt plaintext correctly",
        "Task test_encrypt_long_2 error",
    )
    def test_encrypt_long_2(self):
        for plaintext, expected_ciphertext in long_examples_encrypted_dict_2.items():
            actual_ciphertext = pset.encrypt(
                plaintext, alphabet, self.shift2, self.magic_number2
            )
            self.assertEqual(
                expected_ciphertext,
                actual_ciphertext,
                f"Incorrect encrypted content. Expected: {expected_ciphertext}, got: {actual_ciphertext}",
            )


############################################################
# part 3.3: decrypt text
############################################################


@testsuite_options(4, 1)
class TestPart3_3(unittest.TestCase):
    """Test Decryption"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        self.shift1 = 2
        self.shift2 = 3
        self.magic_number1 = 8
        self.magic_number2 = 10

    @case_options(
        1,
        "Your code does not decrypt correctly",
        "Task test_decrypt_short_1 error",
    )
    def test_decrypt_short_1(self):
        for plaintext, expected_ciphertext in short_examples_encrypted_dict_1.items():
            actual_decrypted_text = pset.decrypt(
                expected_ciphertext, alphabet, self.shift1, self.magic_number1
            )
            self.assertEqual(
                plaintext,
                actual_decrypted_text,
                f"Incorrect decrypted content. Expected: {plaintext}, got: {actual_decrypted_text}",
            )

    @case_options(
        1,
        "Your code does not decrypt correctly",
        "Task test_decrypt_short_2 error",
    )
    def test_decrypt_short_2(self):
        for plaintext, expected_ciphertext in short_examples_encrypted_dict_2.items():
            actual_decrypted_text = pset.decrypt(
                expected_ciphertext, alphabet, self.shift2, self.magic_number2
            )
            self.assertEqual(
                plaintext,
                actual_decrypted_text,
                f"Incorrect decrypted content. Expected: {plaintext}, got: {actual_decrypted_text}",
            )

    @case_options(
        1,
        "Your code does not decrypt correctly",
        "Task test_decrypt_short_3 error",
    )
    def test_decrypt_short_3(self):
        actual_decrypted_text = pset.decrypt("iridescent", alphabet, 0, 1000)
        self.assertEqual(
            "iridescent",
            actual_decrypted_text,
            f"Incorrect decrypted content. Expected: iridescent, got: {actual_decrypted_text}",
        )

    @case_options(
        1,
        "Your code does not decrypt correctly",
        "Task test_decrypt_long_1 error",
    )
    def test_decrypt_long_1(self):
        for plaintext, expected_ciphertext in long_examples_encrypted_dict_1.items():
            actual_decrypted_text = pset.decrypt(
                expected_ciphertext, alphabet, self.shift1, self.magic_number1
            )
            self.assertEqual(
                plaintext,
                actual_decrypted_text,
                f"Incorrect decrypted content. Expected: {plaintext}, got: {actual_decrypted_text}",
            )

    @case_options(
        1,
        "Your code does not decrypt correctly",
        "Task test_decrypt_long_2 error",
    )
    def test_decrypt_long_2(self):
        for plaintext, expected_ciphertext in long_examples_encrypted_dict_2.items():
            actual_decrypted_text = pset.decrypt(
                expected_ciphertext, alphabet, self.shift2, self.magic_number2
            )
            self.assertEqual(
                plaintext,
                actual_decrypted_text,
                f"Incorrect decrypted content. Expected: {plaintext}, got: {actual_decrypted_text}",
            )


############################################################
# part 4.1: count words
############################################################


@testsuite_options(4, 1)
class TestPart4_1(unittest.TestCase):
    """Test Count Words"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        1,
        "Your code does not count words correctly",
        "Task test_count_words error",
    )
    def test_count_words(self):
        for text, expected_count in count_words_dict.items():
            actual_count = pset.count_words(text)
            self.assertEqual(
                expected_count,
                actual_count,
                f"Incorrect next word. Expected: {expected_count}, got: {actual_count}",
            )


############################################################
# part 4.2: break cipher
############################################################


@testsuite_options(10, 1)
class TestPart4_2(unittest.TestCase):
    """Test Break Cipher"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        2.5,
        "Your code does not break cipher correctly",
        "Task test_break_cipher_short error",
    )
    def test_break_cipher_short(self):
        for plaintext, ciphertext in short_examples_encrypted_dict_1.items():
            most_deciphered_text = pset.break_cipher(
                ciphertext, alphabet
            )
            self.assertIn(
                most_deciphered_text,
                short_possible_deciphered_txts_dict[ciphertext],
                f"Failed to break cipher. Expected text in: {short_possible_deciphered_txts_dict[ciphertext]}, got: {most_deciphered_text}",
            )

    @case_options(
        2.5,
        "Your code does not break cipher correctly",
        "Task test_break_cipher_long error",
    )
    def test_break_cipher_long(self):
        for plaintext, ciphertext in long_examples_encrypted_dict_1.items():
            most_deciphered_text = pset.break_cipher(
                ciphertext, alphabet
            )
            self.assertIn(
                most_deciphered_text,
                long_possible_deciphered_txts_dict[ciphertext],
                f"Failed to break cipher. Expected text in: {long_possible_deciphered_txts_dict[ciphertext]}, got: {most_deciphered_text}",
            )


############################################################
# test results reporting
############################################################


class Results_600(unittest.TextTestResult):
    """Custom test result class to capture output and points."""

    def __init__(self, *args, **kwargs):
        super(Results_600, self).__init__(*args, **kwargs)
        self.output = []
        self.points = 0
        self.max_points = 0

    def _getOptions(self, test):
        method_name = getattr(test, "_testMethodName")
        method = getattr(test, method_name)
        func = method.__func__
        points = getattr(func, "points", 0)
        failure_msg = getattr(func, "failure_message", "")
        error_msg = getattr(func, "error_message", "")
        return points, failure_msg, error_msg

    def addSuccess(self, test):
        points, _, _ = self._getOptions(test)
        self.points += points
        self.max_points += points
        return super().addSuccess(test)

    def addFailure(self, test, err):
        points, failure_msg, _ = self._getOptions(test)
        self.output.append(f"❌ [-{points}] {failure_msg}, {err[1]}\n")
        self.max_points += points
        super().addFailure(test, err)

    def addError(self, test, err):
        points, _, error_msg = self._getOptions(test)
        self.output.append(f"❌ [-{points}] {error_msg}, {err[1]}\n")
        self.max_points += points
        super().addError(test, err)

    def getOutput(self):
        """Return the captured output."""
        if self.points > 0:
            self.output.append(
                f"\n✅ [+{self.points}] "
                f"{'All' if self.points == self.max_points else 'Some'}"
                f" tests passed!\n"
            )
        return "\n".join(self.output)

    def getPoints(self):
        """Return the total points."""
        return self.points


if __name__ == "__main__":
    test_parts = [
        TestPart3_1,
        TestPart3_2,
        TestPart3_3,
        TestPart4_1,
        TestPart4_2,
    ]

    suite = unittest.TestSuite()
    for part in test_parts:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(part))
    runner = unittest.TextTestRunner(resultclass=Results_600, verbosity=2)
    result = runner.run(suite)

    output = result.getOutput()
    print(output)
    points_earned = round(result.getPoints(), 3)
    print(f"Total points: {points_earned} / {result.max_points}")
