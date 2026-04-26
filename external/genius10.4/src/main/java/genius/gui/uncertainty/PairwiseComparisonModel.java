package genius.gui.uncertainty;

import java.util.stream.IntStream;

import javax.swing.JOptionPane;

import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.listener.Listener;
import genius.core.repository.ProfileRepItem;
import genius.core.uncertainty.UNCERTAINTYTYPE;
import genius.core.uncertainty.UncertainPreferenceContainer;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.UncertainAdditiveUtilitySpace;
import genius.gui.panels.BooleanModel;
import genius.gui.panels.TextModel;

public class PairwiseComparisonModel {

	private ProfileRepItem profile;

	private TextModel errorModel = new TextModel("0.0");

	private int numberOfComparisons;
	private final int maxNumberOfComparisons;

	private BooleanModel experimentalModel = new BooleanModel(true);
	private BooleanModel confirmationModel = new BooleanModel(false);

	private UncertainPreferenceContainer uncertainPrefContainer = null;

	public PairwiseComparisonModel(ProfileRepItem profile) {
		this.profile = profile;
		this.maxNumberOfComparisons = calculateMaxNumberOfComparisons();
		connect();
	}

	private int calculateMaxNumberOfComparisons() {
		SortedOutcomeSpace realSortedOutcomeSpace = new SortedOutcomeSpace(
				(AbstractUtilitySpace) profile.create());
		return realSortedOutcomeSpace.getAllOutcomes().size() - 1;
	}

	/**
	 * All possible values the "number of comparisons" slider can take.
	 */
	public Integer[] getPossibleValues() {
		if (maxNumberOfComparisons <= 10)
			return (Integer[]) IntStream.rangeClosed(1, maxNumberOfComparisons)
					.boxed().toArray();
		else if (maxNumberOfComparisons <= 100)
			return new Integer[] { 1, 5, 10, maxNumberOfComparisons };
		else
			return new Integer[] { 1, 5, 10, 50, 100, maxNumberOfComparisons };
	}

	public void connect() {

		confirmationModel.addListener(new Listener<Boolean>() {
			@Override
			public void notifyChange(Boolean data) {
				if (confirmationModel.getValue() == true) {
					uncertainPrefContainer = createContainer();
					int amountOfComps = uncertainPrefContainer
							.getPairwiseCompUserModel().getBidRanking()
							.getBidOrder().size() - 1;
					double certaintyLevel = ((double) amountOfComps
							/ (double) getMaxNumberInComps());
					double errorRate = Double
							.parseDouble(getErrorModel().getText()) * 100;

					JOptionPane.showMessageDialog(null,
							"Pairwise Comparison User Model Created Successfully! \n\n"
									+ "Number of Comparisons: " + amountOfComps
									+ "\n" + "Level Of certainty: "
									+ (Math.round((certaintyLevel) * 100)
											/ 100D) * 100
									+ "%\n" + "Error Rate:  "
									+ (Math.round((errorRate) * 100) / 100D)
									+ "%");
				}
			}
		});
	}

	public ProfileRepItem getProfile() {
		return profile;
	}

	public UncertainPreferenceContainer createContainer() {
		UncertainPreferenceContainer container;
		container = new UncertainPreferenceContainer(
				(UncertainAdditiveUtilitySpace) profile.create(),
				UNCERTAINTYTYPE.PAIRWISECOMP);
		return container;
	}

	public TextModel getErrorModel() {
		return errorModel;
	}

	public int getMaxNumberInComps() {
		return maxNumberOfComparisons;
	}

	public BooleanModel getConfirmationModel() {
		return confirmationModel;
	}

	public UncertainPreferenceContainer getUncertainPrefContainer() {
		return uncertainPrefContainer;
	}

	public BooleanModel getExperimentalModel() {
		return experimentalModel;
	}

	public int getNumberOfComparisons() {
		return numberOfComparisons;
	}

	public void setNumberOfComparisons(int numberOfCOmparisons) {
		this.numberOfComparisons = numberOfCOmparisons;
	}

}
