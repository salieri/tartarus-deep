import _ from 'lodash';

import {
  DeferredInputFeed,
  EndOfStreamException,
  InputFeedRecord,
  Sample,
  Label,
} from './feed';

import { NDArray, NDArrayCollection } from '../math';
import { DeferredInputCollection, DeferredCollection } from '../nn';


export type ProtoSampleData = NDArray|NDArrayCollection;

export interface ProtoSample {
  x: ProtoSampleData;
  y?: ProtoSampleData;
}


export class DeferredMemoryInputFeed extends DeferredInputFeed {
  protected offsetPtr: number = 0;

  protected readonly samples: Sample[];

  protected readonly labels?: Label[];

  public constructor(samples: Sample[], labels?: Label[]) {
    super();

    if ((labels) && (samples.length !== labels.length)) {
      throw new Error('Sample and label lengths must match');
    }

    if (samples.length <= 0) {
      throw new Error('Empty sample set');
    }

    this.samples = samples;
    this.labels = labels;
  }


  public count(): number {
    return this.samples.length;
  }


  public offset(): number {
    return this.offsetPtr;
  }


  public async next(): Promise<InputFeedRecord> {
    const offsetPtr = this.offsetPtr;

    if (offsetPtr >= this.samples.length) {
      throw new EndOfStreamException();
    }

    const data: InputFeedRecord = {
      name: this.samples[offsetPtr].name || `sample-${offsetPtr}`,
      sample: this.samples[offsetPtr],
      label: this.labels ? this.labels[offsetPtr] : undefined,
      offset: offsetPtr,
      count: this.count(),
    };

    this.offsetPtr += 1;

    return data;
  }


  public async seek(offset: number): Promise<void> {
    if ((offset < 0) || (offset >= this.count())) {
      throw new Error('Offset out of range');
    }

    if (!(Number.isInteger(offset))) {
      throw new Error('Invalid offset');
    }

    this.offsetPtr = offset;
  }


  public hasLabels(): boolean {
    return !!this.labels;
  }


  protected static prepareCollection(sampleData: ProtoSampleData): DeferredInputCollection {
    const sampleCollection = new DeferredCollection();

    if (sampleData instanceof NDArray) {
      sampleCollection.declareDefault(sampleData.getDims());
      sampleCollection.setDefaultValue(sampleData);
    } else {
      _.each(
        sampleData,
        (val: NDArray, key: string) => {
          sampleCollection.declare(key, val.getDims());
          sampleCollection.setValue(key, val);
        },
      );
    }

    return new DeferredInputCollection(sampleCollection);
  }


  public static factory(protoSamples: ProtoSample[]): DeferredMemoryInputFeed {
    const samples: Sample[] = [];
    const labels: Label[] = [];

    _.each(
      protoSamples,
      (proto: ProtoSample) => {
        samples.push({ raw: DeferredMemoryInputFeed.prepareCollection(proto.x) });

        if (proto.y) {
          labels.push({ raw: DeferredMemoryInputFeed.prepareCollection(proto.y) });
        }
      },
    );

    return new DeferredMemoryInputFeed(samples, (labels.length > 0) ? labels : undefined);
  }
}
