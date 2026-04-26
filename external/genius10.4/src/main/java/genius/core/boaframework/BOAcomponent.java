package genius.core.boaframework;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import genius.core.boaframework.repository.BOAagentRepository;
import genius.core.boaframework.repository.BOArepItem;
import genius.core.exceptions.InstantiateException;

/**
 * Creates a BOA component consisting of the classname of the component, the
 * type of the component, and all parameters. FIXME this creates nothing. It
 * seems just to contain info that can be used to createFrom a BOA component.
 */
public class BOAcomponent implements Serializable {

	private static final long serialVersionUID = 9055936213274664445L;
	/** Classname of the component */
	private String classname;
	/** Type of the component, for example "as" for acceptance condition */
	private BoaType type;
	/**
	 * Parameters which should be used to initialize the component upon creation
	 */
	private HashMap<String, Double> parametervalues;

	/**
	 * Creates a BOA component consisting of the classname of the components,
	 * the type, and the parameters with which the component should be loaded.
	 * 
	 * @param classname
	 *            of the component. Note, this is not checked at all. We now
	 *            also accept absolute file path to a .class file.
	 * @param type
	 *            of the component (for example bidding strategy).
	 * @param values
	 *            parameters of the component.
	 */
	public BOAcomponent(String classname, BoaType type, HashMap<String, Double> values) {
		if (values == null) {
			throw new NullPointerException("values==null");
		}
		this.classname = classname;
		this.type = type;
		this.parametervalues = values;
	}

	/**
	 * Variant of the main constructor in which it is assumed that the component
	 * has no parameters.
	 * 
	 * @param classname
	 *            of the component. Note, this is not checked at all. We now
	 *            also accept absolute file path to a .class file.
	 * @param type
	 *            of the component (for example bidding strategy).
	 */
	public BOAcomponent(String classname, BoaType type) {
		this.classname = classname;
		this.type = type;
		this.parametervalues = new HashMap<String, Double>();
	}

	/**
	 * Add a parameter to the set of parameters of this component.
	 * 
	 * @param name
	 *            of the parameter.
	 * @param value
	 *            of the parameter.
	 */
	public void addParameter(String name, Double value) {
		parametervalues.put(name, value);
	}

	/**
	 * @return name of the class of the component.
	 */
	public String getClassname() {
		return classname;
	}

	/**
	 * @return type of the component.
	 */
	public BoaType getType() {
		return type;
	}

	/**
	 * @return parameters of the component.
	 */
	public HashMap<String, Double> getParameters() {
		return decreaseAccuracy(parametervalues);
	}

	/**
	 * @return original parameters as specified in the GUI.
	 */
	public HashMap<String, Double> getFullParameters() {
		return parametervalues;
	}

	private HashMap<String, Double> decreaseAccuracy(HashMap<String, Double> parameters) {
		Iterator<Entry<String, Double>> it = parameters.entrySet().iterator();
		HashMap<String, Double> map = new HashMap<String, Double>();
		while (it.hasNext()) {
			Map.Entry<String, Double> pairs = (Entry<String, Double>) it.next();
			map.put(pairs.getKey(), pairs.getValue().doubleValue());
		}
		return map;
	}

	public String toString() {
		String params = "";
		if (parametervalues.size() > 0) {
			ArrayList<String> keys = new ArrayList<String>(parametervalues.keySet());
			Collections.sort(keys);
			params = "{";
			for (int i = 0; i < keys.size(); i++) {
				// use doubleValue to keep #digits in string lower
				params += keys.get(i) + "=" + parametervalues.get(keys.get(i)).doubleValue();
				if (i < keys.size() - 1) {
					params += ", ";
				}
			}
			params += "}";
		}
		String shortType = "unknown";
		if (type != null) {
			switch (type) {
			case BIDDINGSTRATEGY:
				shortType = "bs";
				break;
			case ACCEPTANCESTRATEGY:
				shortType = "as";
				break;
			case OPPONENTMODEL:
				shortType = "om";
				break;
			case OMSTRATEGY:
				shortType = "oms";
				break;
			}
		}
		return shortType + ": " + classname + " " + params;
	}

	/**
	 * @return the original parameters from (a temporary instance of) the actual
	 *         component.
	 * @throws InstantiateException
	 *             if repitem can't be loaded
	 */
	public Set<BOAparameter> getOriginalParameters() throws InstantiateException {
		return getRepItem().getInstance().getParameterSpec();
	}

	/**
	 * @return Find back this in the repository.
	 */
	private BOArepItem getRepItem() {
		// CHECK why don't we use the repository item all along?
		BOAagentRepository repo = BOAagentRepository.getInstance();
		switch (type) {
		case ACCEPTANCESTRATEGY:
			return repo.getAcceptanceStrategyRepItem(classname);
		case BIDDINGSTRATEGY:
			return repo.getBiddingStrategyRepItem(classname);
		case OMSTRATEGY:
			return repo.getOpponentModelStrategyRepItem(classname);
		case OPPONENTMODEL:
			return repo.getOpponentModelRepItem(classname);
		default:
			throw new IllegalStateException("BOAcomponent with unknown type encountered:" + type);
		}
	}
}