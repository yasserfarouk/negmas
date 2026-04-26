package genius.gui.session;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.DefaultListModel;
import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

import genius.core.AgentID;
import genius.core.listener.DefaultListenable;
import genius.core.persistent.PersistentDataType;
import genius.core.repository.DomainRepItem;
import genius.core.repository.MultiPartyProtocolRepItem;
import genius.core.repository.PartyRepItem;
import genius.core.session.MultilateralSessionConfiguration;
import genius.core.session.Participant;
import genius.core.session.SessionConfiguration;
import genius.gui.deadline.DeadlineModel;
import genius.gui.negosession.ContentProxy;
import genius.gui.panels.BooleanModel;
import genius.gui.panels.SingleSelectionModel;
import genius.gui.panels.TextModel;

/**
 * Model that stores single session info for MVC.
 */
public class SessionModel extends DefaultListenable<MultilateralSessionConfiguration> {

	// models are all final, as they will be used to hook up the GUI.
	private final SingleSelectionModel<MultiPartyProtocolRepItem> protocolModel;
	private final SingleSelectionModel<DomainRepItem> domainModel;
	private final DeadlineModel deadlineModel = new DeadlineModel();

	// currently selected participant
	private final ParticipantModel participantModel;
	// the list of participants in the nego session
	private final DefaultListModel<Participant> participantsModel = new DefaultListModel<>();

	// selected mediator. Not used if protocol does not need mediator.
	private final SingleSelectionModel<PartyRepItem> mediatorModel;
	private final TextModel mediatorIdModel = new TextModel("Mediator");

	// bilateral-only options
	private final BooleanModel showProgressChart = new BooleanModel(true);
	private final BooleanModel bilateralShowUtilUtilPlot = new BooleanModel(true);
	private final BooleanModel bilateralShowAllBids = new BooleanModel(true);
	private final BooleanModel printEnabled = new BooleanModel(true);
	private final SingleSelectionModel<PersistentDataType> persistentDatatypeModel = new SingleSelectionModel<PersistentDataType>(
			Arrays.asList(PersistentDataType.values()));

	public SessionModel() {
		protocolModel = new SingleSelectionModel<>(ContentProxy.fetchProtocols());
		domainModel = new SingleSelectionModel<>(ContentProxy.fetchDomains());
		participantModel = new ParticipantModel(protocolModel, domainModel);
		mediatorModel = new SingleSelectionModel<PartyRepItem>(new ArrayList<PartyRepItem>());

		updateSubmodels();
		addConstraints();
	}

	/**
	 * @return the current single session run configuration
	 */
	public MultilateralSessionConfiguration getConfiguration() {
		List<Participant> participants = new ArrayList<>();

		MultiPartyProtocolRepItem protocol = (MultiPartyProtocolRepItem) protocolModel.getSelectedItem();

		Participant mediator = null;
		if (protocol.getHasMediator()) {
			// HACK #1463 use profile of participant 1 to avoid null profile.
			mediator = new Participant(new AgentID(mediatorIdModel.getText()),
					(PartyRepItem) mediatorModel.getSelectedItem(), participantsModel.get(0).getProfile());
		}
		for (int n = 0; n < participantsModel.size(); n++) {
			participants.add(participantsModel.getElementAt(n));
		}
		return new SessionConfiguration(protocol, mediator, participants, deadlineModel.getDeadline(),
				persistentDatatypeModel.getSelection());
	}

	/**
	 * @return the protocol data model.
	 */
	public SingleSelectionModel<MultiPartyProtocolRepItem> getProtocolModel() {
		return protocolModel;
	}
	
	public SingleSelectionModel<DomainRepItem> getDomainModel() {
		return domainModel;
	}

	/**
	 * @return the {@link TextModel} for teh Mediator ID
	 */
	public TextModel getMediatorIdModel() {
		return mediatorIdModel;
	}

	/**
	 * @return the mediator model
	 */
	public SingleSelectionModel<PartyRepItem> getMediatorModel() {
		return mediatorModel;
	}

	public DeadlineModel getDeadlineModel() {
		return deadlineModel;
	}

	public SingleSelectionModel<PersistentDataType> getPersistentDatatypeModel() {
		return persistentDatatypeModel;
	}

	/**
	 * @return the settings for the 'participant information'. This setting is
	 *         used when the user clicks the 'add' button.
	 */
	public ParticipantModel getParticipantModel() {
		return participantModel;
	}

	/**
	 * Call this when model is completed (user clicked 'start'). TODO check that
	 * the model is indeed complete.
	 */
	public void modelIsComplete() {
		notifyChange(getConfiguration());
	}

	/**
	 * @return model holding the list of participants.
	 */
	public DefaultListModel<Participant> getParticipantsModel() {
		return participantsModel;
	}

	/**
	 * 
	 * @return boolean whether to show all bid points in the util-util graph.
	 *         Only has meaning if {@link #getBilateralUtilUtilPlot()} is true
	 */
	public BooleanModel getBilateralShowAllBids() {
		return bilateralShowAllBids;
	}

	/**
	 * @return A booleanModel containing a boolean. If true, a util-util graph
	 *         instead of a default graph will be plotted. util-util graph is a
	 *         graph with side A utilities on the X-axis and side B utilities on
	 *         the Y axis. This visualization method allows us to also plot
	 *         other points like the Pareto Frontier and Nash point.
	 * 
	 */
	public BooleanModel getBilateralUtilUtilPlot() {
		return bilateralShowUtilUtilPlot;
	}

	/**
	 * a booleanModel holding true iff user asked to enable print to stdout.
	 * 
	 * @return
	 */
	public BooleanModel getPrintEnabled() {
		return printEnabled;
	}

	/**
	 * 
	 * @return true iff a session progress chart should be shown.
	 */
	public BooleanModel getShowChart() {
		return showProgressChart;
	}

	/****************************** support funcs ******************/
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

	}

	/**
	 * Update the models after a protocol change.
	 * 
	 * @param protocol
	 *            the new protocol.
	 */
	private void updateSubmodels() {
		mediatorModel.setAllItems(ContentProxy.fetchMediatorsForProtocol(protocolModel.getSelection()));
	}

}
