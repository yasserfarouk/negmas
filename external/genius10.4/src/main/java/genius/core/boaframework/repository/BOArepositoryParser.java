package genius.core.boaframework.repository;

import java.util.HashMap;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import genius.core.boaframework.BoaType;

/**
 * Simple XML parser which parses the BOA repository and stores the information
 * for easy access.
 * 
 * @author Mark Hendrikx
 */
class BOArepositoryParser extends DefaultHandler {
	/** List of bidding strategies in the repository */
	HashMap<String, BOArepItem> biddingStrategies = new HashMap<String, BOArepItem>();
	/** List of acceptance strategies in the repository */
	HashMap<String, BOArepItem> acceptanceConditions = new HashMap<String, BOArepItem>();
	/** List of opponent models in the repository */
	HashMap<String, BOArepItem> opponentModels = new HashMap<String, BOArepItem>();
	/** List of opponent model strategies in the repository */
	HashMap<String, BOArepItem> omStrategies = new HashMap<String, BOArepItem>();

	private BOArepItem currentItem;

	/**
	 * Main method used to parse the repository.
	 * 
	 * @param nsURI
	 *            of the XML element.
	 * @param strippedName
	 *            of the XML element.
	 * @param tagName
	 *            of the XML element.
	 * @param attributes
	 *            of the XML element.
	 */
	public void startElement(String nsURI, String strippedName, String tagName,
			Attributes attributes) throws SAXException {
		if (tagName.equals("biddingstrategy")) {
			currentItem = new BOArepItem(attributes.getValue(0),
					attributes.getValue(1), BoaType.BIDDINGSTRATEGY);
		} else if (tagName.equals("acceptancecondition")) {
			currentItem = new BOArepItem(attributes.getValue(0),
					attributes.getValue(1), BoaType.ACCEPTANCESTRATEGY);
		} else if (tagName.equals("opponentmodel")) {
			currentItem = new BOArepItem(attributes.getValue(0),
					attributes.getValue(1), BoaType.OPPONENTMODEL);
		} else if (tagName.equals("omstrategy")) {
			currentItem = new BOArepItem(attributes.getValue(0),
					attributes.getValue(1), BoaType.OMSTRATEGY);
		}
		// else {
		// if (tagName.equals("parameter")) {
		// currentItem.addParameter(new BOAparameter(attributes.getValue(0),
		// new BigDecimal(attributes.getValue(1)),
		// attributes.getValue(2)));
		// }
		// }
	}

	/**
	 * Method which switches the state of the parser if a section has ended.
	 * 
	 * @param nsURI
	 *            of the XML element.
	 * @param strippedName
	 *            of the XML element.
	 * @param tagName
	 *            of the XML element.
	 */
	public void endElement(String nsURI, String strippedName, String tagName)
			throws SAXException {

		if (tagName.equals("biddingstrategy")) {
			biddingStrategies.put(currentItem.getName(), currentItem);
		} else if (tagName.equals("acceptancecondition")) {
			acceptanceConditions.put(currentItem.getName(), currentItem);
		} else if (tagName.equals("opponentmodel")) {
			opponentModels.put(currentItem.getName(), currentItem);
		} else if (tagName.equals("omstrategy")) {
			omStrategies.put(currentItem.getName(), currentItem);
		}
	}

	/**
	 * @return bidding strategies in the BOA repository.
	 */
	public HashMap<String, BOArepItem> getBiddingStrategies() {
		return biddingStrategies;
	}

	/**
	 * @return acceptance strategies in the BOA repository.
	 */
	public HashMap<String, BOArepItem> getAcceptanceConditions() {
		return acceptanceConditions;
	}

	/**
	 * @return opponent models in the BOA repository.
	 */
	public HashMap<String, BOArepItem> getOpponentModels() {
		return opponentModels;
	}

	/**
	 * @return opponent model strategies in the BOA repository.
	 */
	public HashMap<String, BOArepItem> getOMStrategies() {
		return omStrategies;
	}
}