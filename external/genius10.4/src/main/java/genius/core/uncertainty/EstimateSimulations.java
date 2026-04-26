package genius.core.uncertainty;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.Domain;
import genius.core.DomainImpl;
import genius.core.bidding.BidDetails;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.UncertainAdditiveUtilitySpace;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import agents.anac.y2019.harddealer.math3.stat.correlation.SpearmansCorrelation;
/**
 * Purely Experimental.
 * This is the class in which we run the simulations on the estimateUtilitySpace functions present in EstimateUtilityLibrary.
 * @author Adel Magra
 */

public class EstimateSimulations {
	
	static String PARTY = "src/test/resources/partydomain/";
	static String LAPTOP = "etc/templates/laptopdomain/";
	static String ITC = "etc/templates/ItexvsCypressDomain/";
	static String OUTFIT = "etc/templates/anac/y2012/Outfit/";
	
	
	/**
	 * Computes the sum of squared error for two utility functions defined on the same domain.
	 * @param estimate: The utility space to compare.
	 * @param baseline: The true utility space be compared to (NOTE: the order of inputs has no importance).
	 * @return squared error of the the utility functions.
	 */
	public static double computeSquaredError(AdditiveUtilitySpace estimate, AdditiveUtilitySpace baseline) throws Exception{
		 //First make sure that estimate and baseline have the same domain.
		if(!estimate.getDomain().equals(baseline.getDomain()))
			 throw new IllegalArgumentException("The utility spaces are not defined on the same domain");
		
		Domain domain = baseline.getDomain();
		long nrOfBids = domain.getNumberOfPossibleBids();
		BidIterator bidIterator = new BidIterator(domain);
		double error = 0;
		while(bidIterator.hasNext()){
			Bid bid = bidIterator.next();
			error += Math.pow(estimate.getUtility(bid)-baseline.getUtility(bid),2);
		}
		return (error/nrOfBids); 
	}
	
	/**
	 * Orders the bids in an array according to their true utilities .
	 * @param utilSpace
	 * @return utilities of the bids of toRank in the order of the true ranking of bids.
	 * @throws Exception 
	 */
	public static double[] estimate_to_rank(AdditiveUtilitySpace baseline, AdditiveUtilitySpace estimate) throws Exception {
		Domain domain = baseline.getDomain();
		SortedOutcomeSpace sortedSpace = new SortedOutcomeSpace(baseline);
		List<BidDetails> sortedBids = sortedSpace.getOrderedList();
		Collections.reverse(sortedBids);
		double[] full_rank = new double[(int) domain.getNumberOfPossibleBids()];
		for(int i=0; i<full_rank.length; i++){
			full_rank[i]= estimate.getUtility(sortedBids.get(i).getBid());
		}
		return full_rank;
	}
	
	
	/**
	 * Converts utilities in a ranked array.
	 */
	public static double[] true_to_rank(AdditiveUtilitySpace baseline){
		Domain domain = baseline.getDomain();
		BidIterator bidIterator = new BidIterator(domain);
		double[] full_rank = new double[(int) domain.getNumberOfPossibleBids()];
		for(int i=0; i<full_rank.length; i++)
			full_rank[i]=baseline.getUtility(bidIterator.next());
		Arrays.sort(full_rank);
		return full_rank;
	}
	
	/**
	 * This function computes the squared error of estimate_kS when obtained from userModel augmented with b.
	 * @param userModel
	 * @param trueSpace
	 * @param bid 
	 * @return Squared Error
	 * @throws Exception 
	 */
	public static double kS_Augmented_SE(UserModel userModel, AdditiveUtilitySpace trueSpace, Bid b) throws Exception {
		User user = new User((UncertainAdditiveUtilitySpace) trueSpace);
		UserModel augmented = user.elicitRank(b, userModel);
		EstimateUtilityLibrary library = new EstimateUtilityLibrary(trueSpace.getDomain(), augmented.getBidRanking());
		AdditiveUtilitySpace kS_Estimate = library.kS_Movement();
		return computeSquaredError(kS_Estimate, trueSpace);
	}
	
	/**
	 * This function returns the bid that when added to the userModel, minimizes the TSErr of the kS_estimate.
	 * Note: Same domain is assumed.
	 * @param userModel
	 * @param trueSpace
	 * @return Bid
	 * @throws Exception 
	 */
	public static Bid kS_ErrorMinimizer(UserModel userModel, AdditiveUtilitySpace trueSpace) throws Exception {
		Domain domain = trueSpace.getDomain();
		BidIterator bidIterator = new BidIterator(domain);
		double minError = 1.1;
		double error = 0;
		//int count = 0;
		Bid minimizer = userModel.bidRanking.getMinimalBid(); //Could be any bid in the bidRanking, this is arbitrary.
		while(bidIterator.hasNext()){
			Bid nextBid = bidIterator.next();
			if (!(userModel.getBidRanking().getBidOrder().contains(nextBid))) {
				//count++;
				error = kS_Augmented_SE(userModel, trueSpace,nextBid);
				if(error < minError) {
					minError = error;
					minimizer = nextBid;
				}
			}
		}
		System.out.println(minError);
		return minimizer;	
	}
	/**
	 * Same as above but for imp
	 */
	
	public static double imp_Augmented_SE(UserModel userModel, AdditiveUtilitySpace trueSpace, Bid b) throws Exception {
		User user = new User((UncertainAdditiveUtilitySpace) trueSpace);
		UserModel augmented = user.elicitRank(b, userModel);
		EstimateUtilityLibrary library = new EstimateUtilityLibrary(trueSpace.getDomain(), augmented.getBidRanking());
		AdditiveUtilitySpace imp_estimate = library.kS_Movement();
		return computeSquaredError(imp_estimate, trueSpace);
	}
	
	public static Bid imp_ErrorMinimizer(UserModel userModel, AdditiveUtilitySpace trueSpace) throws Exception {
		Domain domain = trueSpace.getDomain();
		BidIterator bidIterator = new BidIterator(domain);
		double minError = 1.1;
		double error = 0;
		//int count = 0;
		Bid minimizer = userModel.bidRanking.getMinimalBid(); //Could be any bid in the bidRanking, this is arbitrary.
		while(bidIterator.hasNext()){
			Bid nextBid = bidIterator.next();
			if (!(userModel.getBidRanking().getBidOrder().contains(nextBid))) {
				//count++;
				error = imp_Augmented_SE(userModel, trueSpace,nextBid);
				if(error < minError) {
					minError = error;
					minimizer = nextBid;
				}
			}
		}
		System.out.println(minError);
		return minimizer;	
	}
	
	/**
	 * Same as above but for default.
	 */
	
	public static double def_Augmented_SE(UserModel userModel, AdditiveUtilitySpace trueSpace, Bid b) throws Exception {
		User user = new User((UncertainAdditiveUtilitySpace) trueSpace);
		UserModel augmented = user.elicitRank(b, userModel);
		EstimateUtilityLibrary library = new EstimateUtilityLibrary(trueSpace.getDomain(), augmented.getBidRanking());
		AdditiveUtilitySpace def_estimate = (AdditiveUtilitySpace) library.default_estimation();
		return computeSquaredError(def_estimate, trueSpace);
	}
	
	public static Bid def_ErrorMinimizer(UserModel userModel, AdditiveUtilitySpace trueSpace) throws Exception {
		Domain domain = trueSpace.getDomain();
		BidIterator bidIterator = new BidIterator(domain);
		double minError = 1.1;
		double error = 0;
		//int count = 0;
		Bid minimizer = userModel.bidRanking.getMinimalBid(); //Could be any bid in the bidRanking, this is arbitrary.
		while(bidIterator.hasNext()){
			Bid nextBid = bidIterator.next();
			if (!(userModel.getBidRanking().getBidOrder().contains(nextBid))) {
				//count++;
				error = def_Augmented_SE(userModel, trueSpace,nextBid);
				if(error < minError) {
					minError = error;
					minimizer = nextBid;
				}
			}
		}
		System.out.println(minError);
		return minimizer;	
	}
	/**
	 * The main function were the simulations are ran.
	 * @param args
	 * @throws Exception 
	 */
	public static void main (String[] args) throws Exception {
		
		/**
		 * T-Tests to test whether or not for a fixed size of bid ranking, TSE_min (after eliciting minimizer) is significantly different 
		 * from TSE_rdm (after eliciting random bid)
		 * Done on LAPTOP domain.
		 */
		DomainImpl domain = new DomainImpl(OUTFIT + "Outfit-A-domain.xml");
		System.out.println(domain.getNumberOfPossibleBids());
		/*DomainImpl domain = new DomainImpl(OUTFIT + "Outfit-A-domain.xml");
		UncertainAdditiveUtilitySpace utilSpace = new UncertainAdditiveUtilitySpace(domain,
				OUTFIT + "Outfit-A-prof1.xml");
		System.out.println(domain.getNumberOfPossibleBids());
		//Bid ranking of size 10.
		double[] error_rdm = new double[30];
		System.out.println("i = 10: SE_Error_Min ");
		for(int i=0; i<30; i++) {
			UncertainPreferenceContainer u = new UncertainPreferenceContainer(utilSpace, UNCERTAINTYTYPE.PAIRWISECOMP, 10);
			UserModel userModel = u.getPairwiseCompUserModel();
			BidRanking bidRank = userModel.getBidRanking();
			UserModel userModel_rdm = new UserModel(bidRank);
			UserModel userModel_min = new UserModel(bidRank);
			Bid minimizer_SE = kS_ErrorMinimizer(userModel_min,utilSpace);
			Bid rdm = bidRank.getRandomBid();
			error_rdm[i] = kS_Augmented_SE(userModel_rdm,utilSpace,rdm);
		}
		
		System.out.println("i = 10: SE_kS_Error_RDM ");
		for(int i = 0; i<30; i++) {
			System.out.println(error_rdm[i]);
		}
		
		System.out.println("i = 40: SE_Error_Min ");
		for(int i=0; i<30; i++) {
			UncertainPreferenceContainer u = new UncertainPreferenceContainer(utilSpace, UNCERTAINTYTYPE.PAIRWISECOMP, 40);
			UserModel userModel = u.getPairwiseCompUserModel();
			BidRanking bidRank = userModel.getBidRanking();
			UserModel userModel_rdm = new UserModel(bidRank);
			UserModel userModel_min = new UserModel(bidRank);
			Bid minimizer_SE = kS_ErrorMinimizer(userModel_min,utilSpace);
			Bid rdm = bidRank.getRandomBid();
			error_rdm[i] = kS_Augmented_SE(userModel_rdm,utilSpace,rdm);
		}
		
		System.out.println("i = 40: SE_kS_Error_RDM ");
		for(int i = 0; i<30; i++) {
			System.out.println(error_rdm[i]);
		}
		
		System.out.println("i = 75: SE_Error_Min ");
		for(int i=0; i<30; i++) {
			UncertainPreferenceContainer u = new UncertainPreferenceContainer(utilSpace, UNCERTAINTYTYPE.PAIRWISECOMP, 75);
			UserModel userModel = u.getPairwiseCompUserModel();
			BidRanking bidRank = userModel.getBidRanking();
			UserModel userModel_rdm = new UserModel(bidRank);
			UserModel userModel_min = new UserModel(bidRank);
			Bid minimizer_SE = kS_ErrorMinimizer(userModel_min,utilSpace);
			Bid rdm = bidRank.getRandomBid();
			error_rdm[i] = kS_Augmented_SE(userModel_rdm,utilSpace,rdm);
		}
		
		System.out.println("i = 75: SE_kS_Error_RDM ");
		for(int i = 0; i<30; i++) {
			System.out.println(error_rdm[i]);
		}
		System.out.println("i = 100: SE_Error_Min ");
		for(int i=0; i<30; i++) {
			UncertainPreferenceContainer u = new UncertainPreferenceContainer(utilSpace, UNCERTAINTYTYPE.PAIRWISECOMP, 100);
			UserModel userModel = u.getPairwiseCompUserModel();
			BidRanking bidRank = userModel.getBidRanking();
			UserModel userModel_rdm = new UserModel(bidRank);
			UserModel userModel_min = new UserModel(bidRank);
			Bid minimizer_SE = kS_ErrorMinimizer(userModel_min,utilSpace);
			Bid rdm = bidRank.getRandomBid();
			error_rdm[i] = kS_Augmented_SE(userModel_rdm,utilSpace,rdm);
		}
		
		System.out.println("i = 100: SE_kS_Error_RDM ");
		for(int i = 0; i<30; i++) {
			System.out.println(error_rdm[i]);
		}
		*/
		/**
		 * Uncomment to compare estimation of kS when eliciting minError vs maxDist.
		 */
		/*System.out.println(domain.getNumberOfPossibleBids());
		//Create a random user model, and simultaneously sequentially add max_TV bid and min_Error.
		UncertainPreferenceContainer u = new UncertainPreferenceContainer(utilSpace, UNCERTAINTYTYPE.PAIRWISECOMP, 4);
		User user = new User(utilSpace);
		UserModel userModel = u.getPairwiseCompUserModel();
		BidRanking bidRank = userModel.getBidRanking();
		UserModel userModel_maxD = new UserModel(bidRank);
		UserModel userModel_min = new UserModel(bidRank);
		System.out.println("def_Error_Min: ");
		double[] error_maxD = new double[23];
		double[] minimizer_dist = new double[23];
		double[] max_dist = new double[23];
		double[] min_dist = new double[23];
		for(int i=0; i<23; i++) {
			Bid minimizer_SE = kS_ErrorMinimizer(userModel_min,utilSpace);
			Bid maximizer_TV = userModel_min.getBidRanking().getTVMaximizer();
			Bid minimizer_TV = userModel_min.getBidRanking().getTVMinimizer();
			minimizer_dist[i] = userModel_min.getBidRanking().addedTV(minimizer_SE);
			max_dist[i] = userModel_min.getBidRanking().addedTV(maximizer_TV);
			min_dist[i] = userModel_min.getBidRanking().addedTV(minimizer_TV);
			error_maxD[i] = kS_Augmented_SE(userModel_maxD,utilSpace,maximizer_TV);
			userModel_min = user.elicitRank(minimizer_SE, userModel_min);
			//userModel_maxD = user.elicitRank(maximizer_TV, userModel_maxD);
		}
		
		System.out.println("def_Error_MaxD: ");
		for(int i = 0; i<23; i++) {
			System.out.println(error_maxD[i]);
		}
		
		System.out.println("MaxDist: ");
		for(int i = 0; i<23; i++) {
			System.out.println(max_dist[i]);
		}
		
		System.out.println("MinimizerDist: ");
		for(int i = 0; i<23; i++) {
			System.out.println(minimizer_dist[i]);
		}
		
		System.out.println("MinimalDist: ");
		for(int i = 0; i<23; i++) {
			System.out.println(min_dist[i]);
		}
		*/
		/**
		 * Un-comment to run tests based on squared error
		 */
		/*//Store the error on estimations of KS and SAGA in two arrays, where i^th index corresponds to an estimation from a bidRank of size i.
				/*double[] kS_Error = new double[47];
				double[] saga_Error = new double[47];
				for(int i=3; i<50; i++) {
					UncertainPreferenceContainer u = new UncertainPreferenceContainer(utilSpace, UNCERTAINTYTYPE.PAIRWISECOMP, i);
					UserModel userModel = u.getPairwiseCompUserModel();
					BidRanking bidRank = userModel.getBidRanking();
					EstimateUtilityLibrary library = new EstimateUtilityLibrary(domain, bidRank);
					AdditiveUtilitySpace estimate_ks = library.kS_Movement();
					AdditiveUtilitySpace estimate_saga = (AdditiveUtilitySpace) library.fitnessCross_SAGA();
					kS_Error[i-3] = computeSquaredError(estimate_ks,utilSpace);
					saga_Error[i-3] = computeSquaredError(estimate_saga,utilSpace);
				}
				
				System.out.println("KS:");
				for(int i = 0 ; i<47; i++) 
					System.out.println(kS_Error[i]);
				System.out.println("SAGA:");
				for(int i = 0 ; i<47; i++)
					System.out.println(saga_Error[i]);
		*/
		
		/**
		 * Uncomment to to run tests based on spearman coefficient.
		 */
		
		//Store the spearman coefficient on estimations of KS and SAGA in two arrays, where i^th index corresponds to an spearman from a bidRank of size i.
		/*double[] true_rank = true_to_rank(utilSpace);
		SpearmansCorrelation spearman_helper = new SpearmansCorrelation();
	 	double[] kS_Spearman = new double[50];
		double[] saga_Spearman = new double[50];
		for(int i=3; i<53; i++) {
			UncertainPreferenceContainer u = new UncertainPreferenceContainer(utilSpace, UNCERTAINTYTYPE.PAIRWISECOMP, i);
			UserModel userModel = u.getPairwiseCompUserModel();
			BidRanking bidRank = userModel.getBidRanking();
			EstimateUtilityLibrary library = new EstimateUtilityLibrary(domain, bidRank);
			AdditiveUtilitySpace estimate_ks = library.kS_Movement();
			AdditiveUtilitySpace estimate_saga = (AdditiveUtilitySpace) library.fitnessCross_SAGA();
			double[] kS_rank = estimate_to_rank(utilSpace,estimate_ks);
			double[] saga_rank = estimate_to_rank(utilSpace,estimate_saga);
			kS_Spearman[i-3] = spearman_helper.correlation(true_rank,kS_rank);
			saga_Spearman[i-3] = spearman_helper.correlation(true_rank,saga_rank);
		}
		
		System.out.println("KS:");
		for(int i = 0 ; i<50; i++) 
			System.out.println(kS_Spearman[i]);
		System.out.println("SAGA:");
		for(int i = 0 ; i<50; i++)
			System.out.println(saga_Spearman[i]);
		*/
		
		/**
		 * Basic comparaison based on squared Error.
		 */
		/*UncertainPreferenceContainer u = new UncertainPreferenceContainer(utilSpace, UNCERTAINTYTYPE.PAIRWISECOMP, 10);
		UserModel userModel = u.getPairwiseCompUserModel();
		BidRanking bidRank = userModel.getBidRanking();
		EstimateUtilityLibrary library = new EstimateUtilityLibrary(domain, bidRank);
		AdditiveUtilitySpace estimate_ks = library.kS_Movement();
		AdditiveUtilitySpace estimate_saga = (AdditiveUtilitySpace) library.fitnessCross_SAGA();
		for (int i=0; i<bidRank.getSize(); i++) {
			Bid bid = bidRank.getBidOrder().get(i);
			System.out.println("estimate_KS: " + estimate_ks.getUtility(bid) + " VS true: " + utilSpace.getUtility(bid) + " VS estimate_SAGA: " + estimate_saga.getUtility(bid));
		}
		System.out.println("Error_KS: " + computeSquaredError(estimate_ks,utilSpace));
		System.out.println("Error_SAGA: " + computeSquaredError(estimate_saga,utilSpace));
		*/
	
	}
		
}
