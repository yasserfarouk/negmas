package genius.gui.tournament;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

import genius.core.config.MultilateralTournamentConfiguration;
import genius.core.listener.DefaultListenable;
import genius.core.listener.Listener;
import genius.core.persistent.PersistentDataType;
import genius.core.repository.MultiPartyProtocolRepItem;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.PartyRepItem;
import genius.core.repository.ProfileRepItem;
import genius.gui.deadline.DeadlineModel;
import genius.gui.negosession.ContentProxy;
import genius.gui.panels.BooleanModel;
import genius.gui.panels.NumberModel;
import genius.gui.panels.SingleSelectionModel;
import genius.gui.panels.SubsetSelectionModel;

/**
 * Contains the basic elements of MultilateralTournamentConfiguration, but
 * mutable and the subcomponents are listenable so that we can use it for the
 * MVC pattern. It listens to changes in the protocol and updates the model when
 * necessary.
 * 
 * <p>
 * You can get notified when the multitournamentmodel is complete (as indicated
 * by the user, he can press 'start' in the GUI). The data passed with the
 * notification is the {@link MultilateralTournamentConfiguration}.
 * 
 * @author W.Pasman
 *
 */
public class MultiTournamentModel
		extends DefaultListenable<MultilateralTournamentConfiguration> {

	// models are all final, as they will be used to hook up the GUI.

	private final SubsetSelectionModel<ProfileRepItem> profileModel;
	private final SubsetSelectionModel<ParticipantRepItem> partyModel;
	private final SingleSelectionModel<MultiPartyProtocolRepItem> protocolModel;
	private final DeadlineModel deadlineModel = new DeadlineModel();
	private final SingleSelectionModel<PartyRepItem> mediatorModel;
	private final NumberModel numTournamentsModel = new NumberModel(1, 1,
			2000000000, 1);
	private final NumberModel numAgentsPerSessionModel = new NumberModel(1, 1,
			2000000000, 1);
	private final BooleanModel agentRepetitionModel = new BooleanModel(false);
	private final BooleanModel randomSessionOrderModel = new BooleanModel(
			false);
	private final BooleanModel enablePrintModel = new BooleanModel(false);
	private final BilateralOptionsModel bilateralOptionsModel;
	private final SingleSelectionModel<PersistentDataType> persistentDatatypeModel = new SingleSelectionModel<PersistentDataType>(
			Arrays.asList(PersistentDataType.values()));

	public MultiTournamentModel() {
		// load initial models.
		protocolModel = new SingleSelectionModel<>(
				ContentProxy.fetchProtocols());
		profileModel = new SubsetSelectionModel<ProfileRepItem>(
				ContentProxy.fetchProfiles());

		// stubs for the partyModel and mediatorModel, will be set properly in
		// updateSubmodels.
		partyModel = new SubsetSelectionModel<ParticipantRepItem>(
				new ArrayList<ParticipantRepItem>());
		mediatorModel = new SingleSelectionModel<PartyRepItem>(
				new ArrayList<PartyRepItem>());

		bilateralOptionsModel = new BilateralOptionsModel(protocolModel);

		updateSubmodels();
		updateAgentRepetition();

		addConstraints();

	}

	public SubsetSelectionModel<ProfileRepItem> getProfileModel() {
		return profileModel;
	}

	/**
	 * @return model containing the deadline information
	 */
	public DeadlineModel getDeadlineModel() {
		return deadlineModel;
	}

	/**
	 * @return model containing the parties to use in the tournament
	 */
	public SubsetSelectionModel<ParticipantRepItem> getPartyModel() {
		return partyModel;
	}

	/**
	 * @return the model containing the number of tournaments to be run. This is
	 *         also called "number of sessions" in some places. May be somethign
	 *         historic.
	 */
	public NumberModel getNumTournamentsModel() {
		return numTournamentsModel;
	}

	/**
	 * @return the model containing the number of agents per session.
	 */
	public NumberModel getNumAgentsPerSessionModel() {
		return numAgentsPerSessionModel;
	}

	/**
	 * @return mediator model, or null if no mediator for this protocol
	 */
	public SingleSelectionModel<PartyRepItem> getMediatorModel() {
		return mediatorModel;
	}

	/**
	 * @return this model, converted in a
	 *         {@link MultilateralTournamentConfiguration}.
	 */
	public MultilateralTournamentConfiguration getConfiguration() {
		List<ProfileRepItem> profilesB = new ArrayList<>();
		List<ParticipantRepItem> partiesB = new ArrayList<>();

		if (!bilateralOptionsModel.getPlayBothSides().getValue()) {
			profilesB = bilateralOptionsModel.getProfileModelB()
					.getSelectedItems();
			partiesB = bilateralOptionsModel.getPartyModelB()
					.getSelectedItems();
		}
		return new MultilateralTournamentConfiguration(
				protocolModel.getSelection(), deadlineModel.getDeadline(),
				mediatorModel.getSelection(), partyModel.getSelectedItems(),
				profileModel.getSelectedItems(), partiesB, profilesB,
				numTournamentsModel.getValue().intValue(),
				numAgentsPerSessionModel.getValue().intValue(),
				agentRepetitionModel.getValue(),
				randomSessionOrderModel.getValue(),
				persistentDatatypeModel.getSelection(),
				enablePrintModel.getValue());

	}

	/**
	 * @return model containing the protocol
	 */
	public SingleSelectionModel<MultiPartyProtocolRepItem> getProtocolModel() {
		return protocolModel;
	}

	public BooleanModel getAgentRepetitionModel() {
		return agentRepetitionModel;
	}

	public BooleanModel getRandomSessionOrderModel() {
		return randomSessionOrderModel;
	}

	/**
	 * Call this when model is completed (user clicked 'start'). TODO check that
	 * the model is indeed complete.
	 */
	public void modelIsComplete() {
		notifyChange(getConfiguration());
	}

	public BilateralOptionsModel getBilateralOptionsModel() {
		return bilateralOptionsModel;
	}

	public SingleSelectionModel<PersistentDataType> getPersistentDatatypeModel() {
		return persistentDatatypeModel;
	}

	/******************* support funcs ***********************/
	/**
	 * Update the repetition setting. Must go to "true, locked" mode if
	 * agentsPerSession is bigger than the number of available agents.
	 */
	private void updateAgentRepetition() {
		agentRepetitionModel.setLock(false);
		if (numAgentsPerSessionModel.getValue().intValue() > partyModel
				.getSelectedItems().size()) {
			agentRepetitionModel.setValue(true);
			agentRepetitionModel.setLock(true);
		}
	}

	/**
	 * connecting listeners that check the constraints between the fields in the
	 * model
	 */
	private void addConstraints() {
		// protocol has major impact on the submodels
		protocolModel.addListDataListener(new ListDataListener() {

			@Override
			public void intervalRemoved(ListDataEvent e) {
			}

			@Override
			public void intervalAdded(ListDataEvent e) {
			}

			@Override
			public void contentsChanged(ListDataEvent e) {
				updateSubmodels();
			}
		});

		// The "Agents per session" field by default equals number of profiles;
		// the agent repetition is depending on this too.
		profileModel.addListener(new Listener<ProfileRepItem>() {
			@Override
			public void notifyChange(ProfileRepItem data) {
				numAgentsPerSessionModel
						.setValue(profileModel.getSelectedItems().size());
				updateAgentRepetition();
			}
		});

		// #parties and numAgentsPerSession -> repetition
		partyModel.addListener(new Listener<ParticipantRepItem>() {
			@Override
			public void notifyChange(ParticipantRepItem data) {
				updateAgentRepetition();
			}
		});
		numAgentsPerSessionModel.addListener(new Listener<Number>() {
			@Override
			public void notifyChange(Number data) {
				updateAgentRepetition();
			}
		});
	}

	/**
	 * Update the models after a protocol change.
	 * 
	 * @param protocol
	 *            the new protocol.
	 */
	private void updateSubmodels() {
		MultiPartyProtocolRepItem protocol = protocolModel.getSelection();
		partyModel.setAllItems(ContentProxy.fetchPartiesForProtocol(protocol));
		mediatorModel
				.setAllItems(ContentProxy.fetchMediatorsForProtocol(protocol));
	}

	public BooleanModel getEnablePrint() {
		return enablePrintModel;
	}

}
