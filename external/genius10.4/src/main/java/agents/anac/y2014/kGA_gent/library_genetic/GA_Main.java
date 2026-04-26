package agents.anac.y2014.kGA_gent.library_genetic;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class GA_Main {

	/**
	 * æ¸¡ã�•ã‚Œã�Ÿé�ºä¼�å­�ã�¨ä¸–ä»£äº¤ä»£ã‚¯ãƒ©ã‚¹ã�«åŸºã�¥ã�„ã�¦é�ºä¼�çš„ã‚¢ãƒ«ã‚´ãƒªã‚ºãƒ ã‚’è¡Œã�†
	 */

	boolean printflag = false;

	List<Gene> list = new ArrayList<Gene>(100);

	Comparator<Gene> comp = new CompGene();;

	GenerationChange generation;

	public GA_Main(Gene gene,GenerationChange generation){
		this.generation = generation;
		list = generation.StartGeneration(gene);
	}
	public GA_Main(Gene gene,GenerationChange generation,Comparator<Gene> comp){

		//System.out.println("call GA_Main instance");
		this.generation = generation;
		list = generation.StartGeneration(gene);
		this.comp = comp;
		//System.out.println("End Making instance");
	}

	public GA_Main(List<Gene> list,GenerationChange generation,Comparator<Gene> comp) {
		// TODO è‡ªå‹•ç”Ÿæˆ�ã�•ã‚Œã�Ÿã‚³ãƒ³ã‚¹ãƒˆãƒ©ã‚¯ã‚¿ãƒ¼ãƒ»ã‚¹ã‚¿ãƒ–
		this.list = list;
		this.comp = comp;
		this.generation = generation;
	}

	public GA_Main(GenerationChange generation,Comparator<Gene> comp){
		//System.out.println("call GA_Main instance");
		this.generation = generation;
		list = generation.StartGeneration();
		this.comp = comp;
		//System.out.println("End Making instance");
	}


	public void Start(){

		int gen  = 0;

		Collections.sort(list, comp);
		Collections.reverse(list);
		//System.out.println("GAStart 0gen");

		do{

			list = generation.Generation(list);
			Collections.sort(list, comp);
			Collections.reverse(list);
			gen++;

			if(printflag){
				System.out.println(gen +" generate " + " maxvalue " +  list.get(0).GetValue());
			}
		}while(generation.End(list)==false);

		if(printflag){
			System.out.println("Generation Change End");
			for (int i = 0; i < 10; i++) {
				System.out.println(list.get(i).GetValue());
			}


		}

	}

	public List<Gene> GetList(){
		return list;
	}


}
