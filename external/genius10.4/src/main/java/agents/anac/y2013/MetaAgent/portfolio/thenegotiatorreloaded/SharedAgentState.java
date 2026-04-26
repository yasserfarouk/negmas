package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

/**
 * When decoupling existing agents into their separate components, it often happens
 * that a component loosely depends on another component; for example an acceptance condition
 * can depend on a target utility calculated by the offering strategy.
 * 
 * To avoid code duplication a Shared Agent State class can be introduced containing the shared
 * code. In this case one of the components calculates the required data, while the other simply
 * requests the stored result.
 * 
 * Note that the only requirement by this class is the implementation of the name. The name
 * should be used to verify that the component which calculates (a part of) the results is
 * available. If this is not the case, the separate component requiring the information calculate
 * the results itself.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public abstract class SharedAgentState {
	
	protected String NAME;
	
	public String getName() {
		return NAME;
	}
}