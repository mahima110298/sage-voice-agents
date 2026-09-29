from rp.numwords import numbers_in_text, spoken_digits, norm


def test_numbers_in_text_digits_and_money():
    assert {"1590", "8281"} <= numbers_in_text("Pay $1,590 to account 8281.")


def test_numbers_in_text_words():
    assert "1590" in numbers_in_text("fifteen hundred ninety dollars")
    assert "200" in numbers_in_text("two hundred dollars")
    assert "1500" in numbers_in_text("one thousand five hundred")


def test_numbers_in_text_digit_sequence():
    assert "7429" in numbers_in_text("my code is seven four two nine")
    assert spoken_digits("seven four two nine") == "7429"
    assert spoken_digits("oh four one") == "041"


def test_norm():
    assert norm("D-5502") == norm("d5502") and norm("May 15th") == norm("may 15")


def test_adjacent_digit_words_not_summed():
    result = numbers_in_text("two hundred dollars to account eight two eight one")
    assert "19" not in result
    assert {"200", "8281"} <= result


def test_twenty_one_still_compound():
    assert "21" in numbers_in_text("twenty one")


def test_decimal_canonical():
    assert numbers_in_text("$1,590.50") == numbers_in_text("$1,590.5")
    assert "1590" in numbers_in_text("1590.00")


def test_no_digit_fragments():
    assert numbers_in_text("two hundred dollars to account eight two eight one") == {"200", "8281"}


def test_all_digit_runs():
    result = numbers_in_text("account eight two eight one, code seven four two nine")
    assert {"8281", "7429"} <= result


def test_teen_then_tens_splits():
    assert numbers_in_text("nineteen eighty five") == {"19", "85"}


def test_one_hundred_twenty():
    assert "120" in numbers_in_text("one hundred twenty")
