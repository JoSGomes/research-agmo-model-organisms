package experiments;

import weka.core.*;
import weka.core.converters.*;
import java.io.*;

import data_processing.ModelOrganism;
import data_processing.Preprocessor;
import weka.core.Instances;
import weka.core.converters.DatabaseSaver;

public class ARFF2Database {

    public static void main(String[] args) throws Exception {
        ModelOrganism[] organism = ModelOrganism.values();
        Preprocessor preprocessor = new Preprocessor(organism[1], 0, "", "KNN");
        Instances dataset = preprocessor.getDatasetsTRAFold();

        DatabaseSaver save = new DatabaseSaver();
        save.setUrl("jdbc:postgresql://localhost:5432/agmo_data");
        save.setUser("postgres");
        save.setPassword("123");
        save.setInstances(dataset);
        save.setRelationForTableName(true);
        save.setTableName("test");
        save.connectToDatabase();
        int count = 0;
        for (int i = 0; i < dataset.numInstances(); i++) {
            save.writeIncremental(dataset.instance(i));
            count++;
            if ((count % 100) == 0)
                System.out.println(count + " rows written so far.");
        }
        // notify saver that we're done
        save.writeIncremental(null);

    }
}
