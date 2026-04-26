package genius.core.misc;

public class StringUtils {
	/**
	 * @param textWithSerial
	 * @return textWithSerial, but with the number at the end of text
	 *         incremented by 1. If textWithSerial does not end on a number, a
	 *         "1" is added to textWithSerial.
	 */
	public static String increment(String textWithSerial) {
		int n = getIndexOfSerialNr(textWithSerial);
		Integer newSerial = 0;
		if (n < textWithSerial.length()) {
			newSerial = Integer.parseInt(textWithSerial.substring(n));
		}
		return textWithSerial.substring(0, n) + (newSerial + 1);
	}

	/**
	 * @return the index of the first digit at the end of the string. All chars
	 *         after this index are digits. Returns text.length() if no serial
	 *         nr at end.
	 */

	static int getIndexOfSerialNr(String txt) {
		int n = txt.length() - 1;
		for (; n >= 0; n--) {
			char c = txt.charAt(n);
			if (c < '0' || c > '9')
				break;
		}
		return n + 1;

	}

}
