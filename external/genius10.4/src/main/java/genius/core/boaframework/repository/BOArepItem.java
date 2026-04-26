package genius.core.boaframework.repository;

import javax.xml.bind.annotation.XmlRootElement;

import genius.core.Global;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.BOA;
import genius.core.boaframework.BoaType;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.exceptions.InstantiateException;

/**
 * Class used to represent an item in the BOArepository. An item in the BOA
 * repository has a classPath and may have a tooltip.
 * 
 * @author Mark Hendrikx
 */
@XmlRootElement
public class BOArepItem implements Comparable<BOArepItem> {
	/** Name of the item */
	private String name;
	/** Classpath of the item in the repository */
	private String classPath;

	private BoaType type;

	public BOArepItem(String name, String classPath, BoaType type) {
		this.name = name;
		this.classPath = classPath;
		this.type = type;
	}

	/**
	 * @return classpath of the BOA component.
	 */
	public String getClassPath() {
		return classPath;
	}

	public String getName() {
		return name;
	}

	public String toString() {
		String output = name + " " + classPath;
		return output;
	}

	public String toXML() {
		String result = "\t\t<";
		String element = "";
		if (type == BoaType.BIDDINGSTRATEGY) {
			element = "biddingstrategy";
		} else if (type == BoaType.ACCEPTANCESTRATEGY) {
			element += "acceptancecondition";
		} else if (type == BoaType.OPPONENTMODEL) {
			element += "opponentmodel";
		} else {
			element += "omstrategy";
		}
		result += element + " description=\"" + name + "\" classpath=\"" + classPath + "\"";
		result += "/>\n";
		return result;
	}

	public BoaType getType() {
		return type;
	}

	public String getTypeString() {
		String result;
		switch (type) {
		case BIDDINGSTRATEGY:
			result = "Bidding strategy";
			break;
		case OPPONENTMODEL:
			result = "Opponent model";
			break;
		case ACCEPTANCESTRATEGY:
			result = "Acceptance strategy";
			break;
		case OMSTRATEGY:
			result = "Opponent model strategy";
			break;
		default:
			result = "Unknown type";
			break;
		}
		return result;
	}

	@Override
	public int compareTo(BOArepItem rep2) {
		if (this.type.ordinal() < rep2.type.ordinal()) {
			return -1; // -1 means that THIS goes before rep2
		}
		if (this.type.ordinal() > rep2.type.ordinal()) {
			return 1;
		}
		if (this.type.ordinal() == rep2.type.ordinal()) {
			return String.CASE_INSENSITIVE_ORDER.compare(this.name, rep2.name);
		}
		return 0;
	}

	/**
	 * @return the {@link BOA} object. This may return a
	 *         {@link OfferingStrategy}, {@link AcceptanceStrategy},
	 *         {@link OpponentModel}, or {@link OMStrategy} depending on the
	 *         type
	 * @throws InstantiateException
	 *             if object can't be loaded
	 */
	public BOA getInstance() throws InstantiateException {
		return (BOA) Global.loadObject(classPath);
	}
}