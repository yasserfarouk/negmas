package genius.core.utility;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import genius.core.Domain;
import genius.core.xml.SimpleDOMParser;
import genius.core.xml.SimpleElement;

/**
 * UncertainAdditiveUtilitySpace is a serializable subset of bids in the bidspace with a
 * full ordering: bid1<=bid2<=bid3...<=bidN. N is <= the total number of bids in
 * the space. Part of the bids may be incorrectly placed depending on the error
 * rate. Instead of containing these bids, this space contains an
 * {@link AdditiveUtilitySpace} from which the N bids are generated.
 */
@SuppressWarnings("serial")
public class UncertainAdditiveUtilitySpace extends AdditiveUtilitySpace {
	private static final String FIXEDSEED = "fixed_seed";
	private static final String EXPERIMENTAL = "experimental";
	private static final String ERROR = "errors";
	private static final String ELICITATION_COST = "elicitation_cost";
	private static final String COMPS = "comparisons";
	
	// DO NOT INITIALIZE THESE FIELDS. AS THE INIT HAPPENS AFTER LOADING
	// SUPERCLASS and our init runs as part of init of superclass
	
	/** number of comparisons in the subset */
	private Integer comparisons;
	
	/** number of bids that are incorrectly placed */
	private Integer errors;
	
	/** The utility it cost to elicit more comparisons */
	private Double elicitationCost;	
	
	/** if true, the random seed for uncertainty remains equal to SEED. */
	private Boolean fixedSeed;
	public static Long SEED = 314L;
	
	/** if true, agents can see the underlying {@link AdditiveUtilitySpace}. */
	private Boolean experimental;

	public UncertainAdditiveUtilitySpace(Domain domain, String fullfile)
			throws IOException {
		super(domain, fullfile);
	}

	/**
	 * Constructor to extend normal AdditiveUtilitySpace to a
	 * UncertainAdditiveUtilitySpace.
	 * 
	 * @param utilitySpace
	 *            the existing AdditiveUtilitySpace.
	 * @param comparisons
	 *            the value for comparisons
	 * @param errors
	 *            the value for errors
	 * @param experimental
	 *            the value for experimental
	 * @param seed
	 *            the value for seed. used to generate the comparisons
	 */
	public UncertainAdditiveUtilitySpace(AdditiveUtilitySpace utilitySpace,
			Integer comparisons, Integer errors, Double elicitationCost, boolean fixedSeed, boolean experimental) {
		super(utilitySpace);
		this.comparisons = comparisons;
		this.errors = errors;
		this.elicitationCost = elicitationCost;
		this.fixedSeed = fixedSeed;
		this.experimental = experimental;
	}

	/**
	 * @return number of comparisons in the user view of this space
	 */
	public Integer getComparisons() {
		return comparisons;
	}

	/**
	 * @return number of errors in the user view of the space
	 */
	public Integer getErrors() {
		return errors;
	}
	
	/**
	 * @return Cost of elicitation actions in the user view of the space
	 */
	public Double getElicitationCost() 
	{
		return elicitationCost;
	}
	
	/**
	 * @return true iff the randomizer with which the comparisons are generated has a fixed seed.
	 */
	public Boolean isFixedSeed() {
		return fixedSeed;
	}

	/**
	 * @return true iff parties are allowed to access the underlying {@link AdditiveUtilitySpace}
	 */
	public Boolean isExperimental() {
		return experimental;
	}

	@Override
	protected boolean loadTreeFromFile(String filename) throws IOException {
		SimpleDOMParser parser = new SimpleDOMParser();
		BufferedReader file = new BufferedReader(
				new FileReader(new File(filename)));
		SimpleElement root = parser.parse(file);

		comparisons = getInt(root, COMPS, 10);
		errors = getInt(root, ERROR, 10);
		elicitationCost = getDouble(root, ELICITATION_COST, 0);
		fixedSeed = getBoolean(root, FIXEDSEED, true);
		experimental = getBoolean(root, EXPERIMENTAL, false);

		return loadTreeRecursive(root);
	}

	@Override
	public SimpleElement toXML() throws IOException {
		SimpleElement root = super.toXML();

		root.setAttribute("type", UTILITYSPACETYPE.UNCERTAIN.toString());
		root.addChildElement(newElement(COMPS, comparisons.toString()));
		root.addChildElement(newElement(ERROR, errors.toString()));
		root.addChildElement(newElement(ELICITATION_COST, elicitationCost.toString()));
		root.addChildElement(newElement(FIXEDSEED, fixedSeed.toString()));
		root.addChildElement(newElement(EXPERIMENTAL, experimental.toString()));
		return root;
	}

	private SimpleElement newElement(String name, String value) {
		SimpleElement element = new SimpleElement(name);
		element.setText(value);
		return element;
	}

	private Integer getInt(SimpleElement currentRoot, String tag,
			int defaultValue) {
		String value = getTag(currentRoot, tag);
		if (value == null) {
			return defaultValue;
		}
		try {
			return Integer.parseInt(value.trim());
		} catch (NumberFormatException e) {
			System.out.println("Failed to read field " + tag);
		}
		return defaultValue;
	}
	
	private Double getDouble(SimpleElement currentRoot, String tag,
			double defaultValue) {
		String value = getTag(currentRoot, tag);
		if (value == null) {
			return defaultValue;
		}
		try {
			return Double.parseDouble(value.trim());
		} catch (NumberFormatException e) {
			System.out.println("Failed to read field " + tag);
		}
		return defaultValue;
	}

	private Long getLong(SimpleElement currentRoot, String tag,
			Long defaultValue) {
		String value = getTag(currentRoot, tag);
		if (value == null) {
			return defaultValue;
		}
		try {
			return Long.parseLong(value.trim());
		} catch (NumberFormatException e) {
			System.out.println("Failed to read field " + tag);
		}
		return defaultValue;
	}

	private Boolean getBoolean(SimpleElement currentRoot, String tag,
			boolean defaultValue) {
		String value = getTag(currentRoot, tag);
		if (value == null) {
			return defaultValue;
		}
		return Boolean.valueOf(value.trim());
	}

	private String getTag(SimpleElement currentRoot, String tagname) {
		Object[] tag = currentRoot.getChildByTagName(tagname);
		if (tag == null || tag.length == 0) {
			return null;
		}
		SimpleElement element = (SimpleElement) (tag[0]);
		return element.getText();
	}

	/**
	 * 
	 * @return the seed to be used for generating the random comparison-elements
	 *         for uncertain profile
	 */
	public Long getSeed() 
	{
		if (fixedSeed)
			return SEED;
		else
			return 0L;
	}

}
