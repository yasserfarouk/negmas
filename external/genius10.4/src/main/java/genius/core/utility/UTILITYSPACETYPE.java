package genius.core.utility;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import genius.core.xml.SimpleDOMParser;
import genius.core.xml.SimpleElement;

public enum UTILITYSPACETYPE {
	LINEAR, NONLINEAR, CONSTRAINT, UNCERTAIN;

	public static UTILITYSPACETYPE getUtilitySpaceType(String filename) {

		SimpleDOMParser parser = new SimpleDOMParser();
		BufferedReader file;
		try {
			file = new BufferedReader(new FileReader(new File(filename)));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		}
		SimpleElement root;
		try {
			root = parser.parse(file);
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		if (root == null)
			return null;
		String attr = root.getAttribute("type");
		if (attr == null) { // FIXME should have proper typing...
			return LINEAR;
		}
		return valueOf(attr.toUpperCase());
		// if (attr.equals("nonlinear"))
		// return NONLINEAR;
		// if (attr.equals("constraint"))
		// return CONSTRAINT;
		// return LINEAR; // FIXME unknown should throw?
	}
}
